"""纯逻辑模块：概率模型、期望间隔、CDF、模拟、注入与判定解析。

本模块不依赖 AstrBot / matplotlib，可直接做单元测试。

概率模型（k 从 1 开始，为"自上次触发后的第 k 条消息"）：
    p_k = min(p_max, p_0 + (p_max - p_0) * (k - 1) / n)

约定：
* p_0、p_max 会被截断到 [0, 1]；若 p_max < p_0，则取 p_max = p_0（概率恒定）。
* n <= 0 表示"0 步爬升"，即从第一条起就是 p_max。
"""
from __future__ import annotations

import bisect
import math
import random
import re
import time
from dataclasses import dataclass, field

MAX_INTERVALS = 5000  # 实测间隔最多保留多少条


# --------------------------------------------------------------------------
# 概率模型
# --------------------------------------------------------------------------
MAX_RAMP_STEPS = 10000          # 上升步数上限，防止超大值拖垮计算
DEFAULT_CUSTOM = (0.05, 0.35, 20)  # 自定义档里某项无效时使用的默认值 (p0, p_max, n)


# --------------------------------------------------------------------------
# 配置值清洗：WebUI / 手改配置文件都可能给出 None、字符串、NaN、越界值
# --------------------------------------------------------------------------
def to_float(v) -> float | None:
    """转成有限小数；None、布尔、非数字、NaN、inf 一律返回 None。"""
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def to_int(v) -> int | None:
    """转成整数；20.0 / "20" 可以，20.5、NaN、布尔、非数字返回 None。"""
    f = to_float(v)
    if f is None or f != int(f):
        return None
    return int(f)


def parse_bool(v, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on", "是", "开"):
            return True
        if s in ("false", "0", "no", "off", "", "否", "关"):
            return False
    return default


def clean_str_list(v) -> list[str]:
    """把配置里的“列表”清洗成非空字符串列表。字符串按行拆分，非字符串/数字的元素丢弃。"""
    if v is None:
        return []
    if isinstance(v, str):
        v = v.splitlines()
    elif not isinstance(v, (list, tuple, set)):
        return []
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (str, int, float)):
            continue
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def sanitize_params(p0, p_max, n) -> tuple[tuple[float, float, int], list[str]]:
    """校验自定义档的三个数字。无效值改用默认值，并返回给用户看的提示。"""
    d0, dm, dn = DEFAULT_CUSTOM
    warns: list[str] = []

    a = to_float(p0)
    if a is None or not 0.0 <= a <= 1.0:
        warns.append(f"初始概率 p0 = {p0!r} 无效（需要 0~1 之间的小数，如 0.05 而不是 5%），已改用 {d0}")
        a = d0
    b = to_float(p_max)
    if b is None or not 0.0 <= b <= 1.0:
        warns.append(f"最高概率 p_max = {p_max!r} 无效（需要 0~1 之间的小数），已改用 {dm}")
        b = dm
    k = to_int(n)
    if k is None or not 0 <= k <= MAX_RAMP_STEPS:
        warns.append(f"上升步数 = {n!r} 无效（需要 0~{MAX_RAMP_STEPS} 的整数），已改用 {dn}")
        k = dn
    if b < a:
        warns.append(f"最高概率 {b:g} 小于初始概率 {a:g}，已把 p_max 调整为 {a:g}（概率恒定）")
        b = a
    return (a, b, k), warns


def normalize(p0: float, p_max: float, n: int) -> tuple[float, float, int]:
    """把参数规整到合法范围（内部兜底；用户输入请先走 sanitize_params）。"""
    d0, dm, dn = DEFAULT_CUSTOM
    p0 = to_float(p0)
    p_max = to_float(p_max)
    n = to_float(n)
    p0 = d0 if p0 is None else min(max(p0, 0.0), 1.0)
    p_max = dm if p_max is None else min(max(p_max, 0.0), 1.0)
    if p_max < p0:
        p_max = p0
    n = dn if n is None else min(max(int(n), 0), MAX_RAMP_STEPS)
    return p0, p_max, n


# 三档预设 (p0, p_max, n)。理论 E[T] 分别约 15.7 / 8.1 / 3.8 条，
# 即每 100 条消息约被注入 6.4 / 12.4 / 26.2 次（模型不一定照做，实际会更低）。
PRESETS: dict[str, tuple[float, float, int]] = {
    "low": (0.02, 0.15, 30),
    "mid": (0.05, 0.35, 20),
    "high": (0.15, 0.70, 10),
}
PRESET_LABELS = {"low": "低", "mid": "中", "high": "高", "custom": "自定义"}


def prob_at(k: int, p0: float, p_max: float, n: int) -> float:
    """第 k 条消息（k>=1）的注入概率 p_k。"""
    p0, p_max, n = normalize(p0, p_max, n)
    k = max(int(k), 1)
    if n == 0:
        return p_max
    return min(p_max, p0 + (p_max - p0) * (k - 1) / n)


def survival(kmax: int, p0: float, p_max: float, n: int) -> list[float]:
    """S(0..kmax)，S(k) = prod_{j=1..k} (1 - p_j)，S(0) = 1。"""
    s = [1.0]
    for j in range(1, kmax + 1):
        s.append(s[-1] * (1.0 - prob_at(j, p0, p_max, n)))
    return s


def cdf(kmax: int, p0: float, p_max: float, n: int) -> list[float]:
    """F(0..kmax)，F(k) = 1 - S(k)。"""
    return [1.0 - x for x in survival(kmax, p0, p_max, n)]


def expected_interval(p0: float, p_max: float, n: int) -> float:
    """平均触发间隔 E[T] = sum_{k>=0} S(k)。

    k >= n+1 时概率恒为 p_max，因此 k >= n+1 的尾项之和有闭式 S(n+1) / p_max：
        E[T] = S(0) + ... + S(n) + S(n+1) / p_max
    p_max = 0 时永远不会触发，返回 inf。
    """
    p0, p_max, n = normalize(p0, p_max, n)
    if p_max <= 0.0:
        return math.inf
    s = survival(n + 1, p0, p_max, n)
    return sum(s[: n + 1]) + s[n + 1] / p_max


def expected_triggers(total: float, p0: float, p_max: float, n: int) -> float:
    """每 total 条消息的预期触发次数 ≈ total / E[T]。"""
    et = expected_interval(p0, p_max, n)
    if math.isinf(et):
        return 0.0
    return total / et


def interval_moments(p0: float, p_max: float, n: int) -> tuple[float, float]:
    """触发间隔 T 的 (均值, 标准差)。p_max = 0 时返回 (inf, inf)。"""
    p0, p_max, n = normalize(p0, p_max, n)
    if p_max <= 0.0:
        return math.inf, math.inf
    s, m1, m2 = 1.0, 0.0, 0.0
    for k in range(1, n + 1):  # 爬升段逐项累加
        p = prob_at(k, p0, p_max, n)
        pk = s * p
        m1 += k * pk
        m2 += k * k * pk
        s *= 1.0 - p
    # k >= n+1 时概率恒为 p_max：T = n + J，J 服从成功率 p_max 的几何分布，尾项用闭式，不再循环
    m1 += s * (n + 1.0 / p_max)
    m2 += s * (n * n + 2.0 * n / p_max + (2.0 - p_max) / (p_max * p_max))
    return m1, math.sqrt(max(m2 - m1 * m1, 0.0))


# --------------------------------------------------------------------------
# 统计：预期 vs 实际、样本量
# --------------------------------------------------------------------------
Z_ALPHA = 1.96    # 双侧 95% 置信
Z_POWER = 0.8416  # 80% 功效
MIN_EXPECTED_TRIGGERS = 30  # 预期触发次数低于它时，检验结论只供参考


def required_triggers(p0: float, p_max: float, n: int, rel_dev: float,
                      z_alpha: float = Z_ALPHA, z_power: float = Z_POWER) -> int:
    """要以给定置信度和功效看出"实际触发数偏离预期 rel_dev（如 0.2 = 20%）"，需要累计的触发次数。
    触发次数的相对波动约为 CV/sqrt(m)，其中 CV = 间隔的标准差/均值。"""
    mean, sd = interval_moments(p0, p_max, n)
    if math.isinf(mean) or rel_dev <= 0:
        return 0
    cv = sd / mean
    return math.ceil(((z_alpha + z_power) * cv / rel_dev) ** 2)


def count_test(messages: int, hits: int, p0: float, p_max: float, n: int) -> dict:
    """把实际触发次数与预期 messages / E[T] 比较（更新过程的正态近似）。
    返回 expected、actual、deviation（相对偏差）、z、p_value（双侧）、reliable。"""
    mean, sd = interval_moments(p0, p_max, n)
    if math.isinf(mean) or messages <= 0:
        return {"expected": 0.0, "actual": hits, "deviation": None,
                "z": None, "p_value": None, "reliable": False}
    expected = messages / mean
    var = messages * sd * sd / mean ** 3
    if var > 1e-12:
        z = (hits - expected) / math.sqrt(var)
    elif abs(hits - expected) < 1e-9:
        z = 0.0
    else:  # 间隔恒定（如每条必注入）：任何偏离都不可能是随机波动
        z = math.copysign(math.inf, hits - expected)
    return {
        "expected": expected,
        "actual": hits,
        "deviation": (hits - expected) / expected if expected > 0 else None,
        "z": z,
        "p_value": math.erfc(abs(z) / math.sqrt(2)),
        "reliable": expected >= MIN_EXPECTED_TRIGGERS,
    }


def wilson_interval(successes: int, total: int, z: float = Z_ALPHA) -> tuple[float, float]:
    """比例的 Wilson 置信区间（小样本下比普通正态区间可靠）。"""
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(center - half, 0.0), min(center + half, 1.0)


def sample_interval(rng: random.Random, probs: list[float], p_max: float,
                    max_k: int = 10_000_000) -> int:
    """模拟一次触发间隔 T。probs[i] 为 p_{i+1}（长度 n+1），之后恒为 p_max。"""
    if p_max <= 0.0:
        return max_k
    k = 1
    m = len(probs)
    rnd = rng.random
    while k <= max_k:
        p = probs[k - 1] if k <= m else p_max
        if rnd() < p:
            return k
        k += 1
    return max_k


def simulate_mean_interval(p0: float, p_max: float, n: int, trials: int,
                           seed: int | None = 0) -> float:
    """蒙特卡洛：模拟 trials 次触发间隔，返回样本均值。"""
    p0, p_max, n = normalize(p0, p_max, n)
    rng = random.Random(seed)
    probs = [prob_at(k, p0, p_max, n) for k in range(1, n + 2)]
    total = 0
    for _ in range(trials):
        total += sample_interval(rng, probs, p_max)
    return total / trials


def empirical_cdf(intervals: list[int], kmax: int) -> list[float]:
    """经验 CDF：F_emp(k) = 样本中 T <= k 的比例，k = 0..kmax。"""
    if not intervals:
        return [0.0] * (kmax + 1)
    xs = sorted(intervals)
    n = len(xs)
    return [bisect.bisect_right(xs, k) / n for k in range(kmax + 1)]


# --------------------------------------------------------------------------
# 会话状态
# --------------------------------------------------------------------------
@dataclass
class SessionState:
    k: int = 1  # 下一条消息是"自上次触发后的第 k 条"

    def to_dict(self) -> dict:
        return {"k": int(self.k)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "SessionState":
        k = to_int(d.get("k", 1)) if isinstance(d, dict) else None
        return cls(k=max(k, 1)) if k is not None else cls()


@dataclass
class RoundInfo:
    """一次请求-响应过程中的临时信息（不持久化）。"""
    umo: str
    k: int
    injected: bool
    ts: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# 注入：只改 req.prompt，绝不碰 system_prompt / contexts
# --------------------------------------------------------------------------
def inject_style(req, instruction: str) -> bool:
    """把风格指令追加到本轮用户消息末尾。返回是否实际注入。"""
    text = (instruction or "").strip()
    if not text:
        return False
    prompt = getattr(req, "prompt", None) or ""
    if prompt.rstrip().endswith(text):
        return False  # 防止重复注入
    req.prompt = f"{prompt}\n\n{text}" if prompt else text
    return True


# --------------------------------------------------------------------------
# 判定相关
# --------------------------------------------------------------------------
DEFAULT_JUDGE_PROMPT = (
    "你是一个文本风格判定器。机器人被要求在回复中做到下面这一点：\n"
    "【{instruction}】\n\n"
    "{user}机器人的回复：\n\"\"\"\n{reply}\n\"\"\"\n\n"
    "请判断这条回复是否确实体现了上述要求。只回答一个字：是 或 否，不要解释。"
)


def build_judge_prompt(template: str, instruction: str, reply: str,
                       user_text: str = "") -> str:
    tpl = (template.strip() if isinstance(template, str) else "") or DEFAULT_JUDGE_PROMPT
    user_block = f"用户的话：\n\"\"\"\n{user_text}\n\"\"\"\n\n" if user_text else ""
    out = (tpl.replace("{instruction}", instruction or "")
              .replace("{user}", user_block)
              .replace("{reply}", reply))
    if "{reply}" not in tpl:  # 模板里没写 {reply}，把回复补在末尾
        out += f"\n\n机器人的回复：\n\"\"\"\n{reply}\n\"\"\""
    return out


_YES = re.compile(r"^\W*(是|对|有|yes|y|true)", re.I)
_NO = re.compile(r"^\W*(否|不|没|无|no|n|false)", re.I)


def parse_yes_no(text: str | None) -> bool | None:
    """解析判定模型的回答。能识别返回 True/False，否则返回 None。"""
    t = (text or "").strip()
    if not t:
        return None
    if _YES.match(t):
        return True
    if _NO.match(t):
        return False
    return None


def keyword_hit(text: str, keywords: list[str] | None) -> bool:
    """任一关键词出现即视为命中（仅用于"确定是"的快速路径）。"""
    if not text or not keywords:
        return False
    return any(kw and kw in text for kw in keywords)
