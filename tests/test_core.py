"""不依赖 AstrBot 的单元测试（只用标准库）。

运行：在插件目录的上一级执行
    python -m unittest discover -s astrbot_plugin_style_prob/tests -v
"""
import copy
import math
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_style_prob import core  # noqa: E402

PARAM_SETS = [
    (0.05, 0.50, 20),
    (0.10, 0.10, 5),    # 恒定概率
    (0.02, 0.30, 50),
    (0.30, 0.90, 3),
    (0.01, 0.20, 100),
]


def pmf_moments(p0, pm, n, kmax=20000):
    """用 P(T=k) = S(k-1) * p_k 直接求和，得到均值、方差、总质量（与闭式独立）。"""
    s_prev, mass, m1, m2 = 1.0, 0.0, 0.0, 0.0
    for k in range(1, kmax + 1):
        p = core.prob_at(k, p0, pm, n)
        pk = s_prev * p
        mass += pk
        m1 += k * pk
        m2 += k * k * pk
        s_prev *= 1 - p
        if s_prev < 1e-15:
            break
    return m1, m2 - m1 * m1, mass


class ProbModel(unittest.TestCase):
    def test_endpoints_and_linear(self):
        p0, pm, n = 0.1, 0.6, 10
        self.assertAlmostEqual(core.prob_at(1, p0, pm, n), p0)
        self.assertAlmostEqual(core.prob_at(6, p0, pm, n), p0 + (pm - p0) * 5 / 10)
        self.assertAlmostEqual(core.prob_at(n + 1, p0, pm, n), pm)
        self.assertAlmostEqual(core.prob_at(n + 50, p0, pm, n), pm)

    def test_degenerate_params(self):
        self.assertAlmostEqual(core.prob_at(1, 0.1, 0.5, 0), 0.5)   # n=0 直接 p_max
        self.assertAlmostEqual(core.prob_at(3, 0.5, 0.2, 10), 0.5)  # p_max<p_0 -> 恒为 p_0
        self.assertTrue(math.isinf(core.expected_interval(0, 0, 10)))
        self.assertAlmostEqual(core.expected_interval(0.3, 1.0, 0), 1.0)

    def test_constant_probability_is_geometric(self):
        for p in (0.05, 0.1, 0.5):
            self.assertAlmostEqual(core.expected_interval(p, p, 7), 1 / p, places=9)

    def test_closed_form_matches_independent_sum(self):
        for p0, pm, n in PARAM_SETS:
            mean, _, mass = pmf_moments(p0, pm, n)
            self.assertAlmostEqual(mass, 1.0, places=9)
            self.assertAlmostEqual(core.expected_interval(p0, pm, n), mean, places=6,
                                   msg=(p0, pm, n))

    def test_cdf_properties(self):
        for p0, pm, n in PARAM_SETS:
            f = core.cdf(300, p0, pm, n)
            self.assertEqual(f[0], 0.0)
            self.assertTrue(all(b >= a for a, b in zip(f, f[1:])))
            s = core.survival(300, p0, pm, n)
            self.assertAlmostEqual(f[10], 1 - s[10])
            # F(k) = 1 - S(k) 也应等于 sum_{j<=k} P(T=j)
            acc, s_prev = 0.0, 1.0
            for j in range(1, 31):
                p = core.prob_at(j, p0, pm, n)
                acc += s_prev * p
                s_prev *= 1 - p
            self.assertAlmostEqual(f[30], acc, places=9)

    def test_expected_triggers(self):
        self.assertAlmostEqual(core.expected_triggers(100, 0.1, 0.1, 5), 10.0)
        self.assertEqual(core.expected_triggers(100, 0, 0, 5), 0.0)


class Presets(unittest.TestCase):
    def test_three_tiers_are_ordered_and_in_expected_range(self):
        et = {k: core.expected_interval(*v) for k, v in core.PRESETS.items()}
        self.assertLess(et["high"], et["mid"])
        self.assertLess(et["mid"], et["low"])
        self.assertAlmostEqual(et["low"], 15.7, delta=0.1)
        self.assertAlmostEqual(et["mid"], 8.1, delta=0.1)
        self.assertAlmostEqual(et["high"], 3.8, delta=0.1)
        for p0, pm, n in core.PRESETS.values():
            self.assertTrue(0 < p0 < pm <= 1 and n > 0)

    def test_preset_monte_carlo(self):
        for p in core.PRESETS.values():
            sim = core.simulate_mean_interval(*p, 200_000, seed=99)
            self.assertAlmostEqual(sim, core.expected_interval(*p),
                                   delta=0.01 * core.expected_interval(*p))


class Statistics(unittest.TestCase):
    def test_interval_moments_match_pmf(self):
        for p0, pm, n in PARAM_SETS:
            mean, sd = core.interval_moments(p0, pm, n)
            m1, var, _ = pmf_moments(p0, pm, n)
            self.assertAlmostEqual(mean, m1, places=6)
            self.assertAlmostEqual(sd, math.sqrt(var), places=6)
            self.assertAlmostEqual(mean, core.expected_interval(p0, pm, n), places=6)
        self.assertTrue(math.isinf(core.interval_moments(0, 0, 5)[0]))

    def test_required_triggers(self):
        mid = core.PRESETS["mid"]
        self.assertEqual([core.required_triggers(*mid, d) for d in (0.3, 0.2, 0.1)], [31, 69, 273])
        # 偏差要求越小，需要的样本越多，且约与 1/dev^2 成正比
        self.assertAlmostEqual(core.required_triggers(*mid, 0.1) / core.required_triggers(*mid, 0.2), 4, delta=0.2)
        self.assertEqual(core.required_triggers(0, 0, 5, 0.2), 0)

    def test_wilson_interval(self):
        lo, hi = core.wilson_interval(8, 10)
        self.assertAlmostEqual(lo, 0.490, delta=0.002)
        self.assertAlmostEqual(hi, 0.943, delta=0.002)
        self.assertEqual(core.wilson_interval(0, 0), (0.0, 1.0))
        lo, hi = core.wilson_interval(10, 10)
        self.assertEqual(hi, 1.0)
        self.assertLess(lo, 1.0)

    def test_count_test_basic(self):
        mean, _ = core.interval_moments(*core.PRESETS["mid"])
        t = core.count_test(1000, round(1000 / mean), *core.PRESETS["mid"])
        self.assertLess(abs(t["z"]), 0.5)
        self.assertGreater(t["p_value"], 0.5)
        self.assertTrue(t["reliable"])
        low = core.count_test(1000, 60, *core.PRESETS["mid"])    # 预期约 124，实际只有 60
        self.assertLess(low["z"], -4)
        self.assertLess(low["p_value"], 0.001)
        self.assertLess(low["deviation"], -0.4)
        self.assertFalse(core.count_test(20, 1, *core.PRESETS["mid"])["reliable"])
        self.assertIsNone(core.count_test(0, 0, *core.PRESETS["mid"])["z"])

    def test_count_test_calibration(self):
        """仿真校准：理想遵循下，z 应近似标准正态；假阳性率约 5%。"""
        p = core.PRESETS["mid"]
        rng = random.Random(2024)
        probs = [core.prob_at(k, *p) for k in range(1, p[2] + 2)]
        N, reps = 2000, 3000
        zs = []
        for _ in range(reps):
            t, hits = 0, 0
            while True:
                t += core.sample_interval(rng, probs, p[1])
                if t > N:
                    break
                hits += 1
            zs.append(core.count_test(N, hits, *p)["z"])
        mean = sum(zs) / reps
        sd = math.sqrt(sum((z - mean) ** 2 for z in zs) / reps)
        fp = sum(abs(z) > 1.96 for z in zs) / reps
        self.assertAlmostEqual(mean, 0.0, delta=0.08)
        self.assertAlmostEqual(sd, 1.0, delta=0.07)
        self.assertAlmostEqual(fp, 0.05, delta=0.02)


class MonteCarlo(unittest.TestCase):
    TRIALS = 300_000

    def test_mean_interval_matches_formula(self):
        for p0, pm, n in PARAM_SETS[:4]:
            theory = core.expected_interval(p0, pm, n)
            _, var, _ = pmf_moments(p0, pm, n)
            sim = core.simulate_mean_interval(p0, pm, n, self.TRIALS, seed=12345)
            tol = max(5 * math.sqrt(var / self.TRIALS), 0.003 * theory)  # 5 个标准误
            self.assertAlmostEqual(sim, theory, delta=tol,
                                   msg=f"{(p0, pm, n)}: sim={sim:.4f} theory={theory:.4f}")

    def test_empirical_cdf_matches_theory(self):
        p0, pm, n = 0.05, 0.5, 20
        rng = random.Random(7)
        probs = [core.prob_at(k, p0, pm, n) for k in range(1, n + 2)]
        samples = [core.sample_interval(rng, probs, pm) for _ in range(100_000)]
        emp = core.empirical_cdf(samples, 60)
        theory = core.cdf(60, p0, pm, n)
        for k in (1, 3, 5, 10, 20, 30, 60):
            self.assertAlmostEqual(emp[k], theory[k], delta=0.01, msg=f"k={k}")


class Injection(unittest.TestCase):
    def test_only_prompt_changes(self):
        req = SimpleNamespace(
            prompt="你好",
            system_prompt="SYSTEM PROMPT",
            contexts=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
            image_urls=[],
        )
        before = copy.deepcopy(req)
        self.assertTrue(core.inject_style(req, "请用撒娇语气"))
        self.assertEqual(req.system_prompt, before.system_prompt)
        self.assertEqual(req.contexts, before.contexts)
        self.assertEqual(req.image_urls, before.image_urls)
        self.assertTrue(req.prompt.startswith(before.prompt))  # 前缀不变，只在末尾追加
        self.assertTrue(req.prompt.endswith("请用撒娇语气"))

    def test_no_double_injection_and_empty(self):
        req = SimpleNamespace(prompt="hi")
        self.assertTrue(core.inject_style(req, "X"))
        self.assertFalse(core.inject_style(req, "X"))
        self.assertFalse(core.inject_style(SimpleNamespace(prompt="hi"), "  "))
        empty = SimpleNamespace(prompt="")
        self.assertTrue(core.inject_style(empty, "X"))
        self.assertEqual(empty.prompt, "X")


class JudgeHelpers(unittest.TestCase):
    def test_parse_yes_no(self):
        for t in ("是", " 是。", "是的", "Yes", "yes."):
            self.assertIs(core.parse_yes_no(t), True, t)
        for t in ("否", "不是", "No", "否，因为……"):
            self.assertIs(core.parse_yes_no(t), False, t)
        for t in ("", None, "也许", "unclear"):
            self.assertIsNone(core.parse_yes_no(t), t)

    def test_keyword_hit(self):
        self.assertTrue(core.keyword_hit("喵喵～", ["喵"]))
        self.assertFalse(core.keyword_hit("你好", ["喵", ""]))
        self.assertFalse(core.keyword_hit("你好", []))

    def test_build_judge_prompt(self):
        p = core.build_judge_prompt("", "撒娇", "回复内容", "用户内容")
        self.assertIn("撒娇", p)
        self.assertIn("回复内容", p)
        self.assertIn("用户内容", p)
        p2 = core.build_judge_prompt("自定义模板，没有占位符", "撒娇", "回复内容")
        self.assertIn("回复内容", p2)  # 缺 {reply} 时自动补在末尾

    def test_state_roundtrip(self):
        self.assertEqual(core.SessionState.from_dict({"k": 7}).k, 7)
        self.assertEqual(core.SessionState.from_dict({"k": "bad"}).k, 1)
        self.assertEqual(core.SessionState.from_dict(None).k, 1)


if __name__ == "__main__":
    unittest.main()
