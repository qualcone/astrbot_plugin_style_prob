"""说话风格概率注入插件。

流程：
  on_llm_request  -> 按 p_k 抽样；命中则把风格指令追加到 req.prompt 末尾
  on_llm_response -> 同步把 k 加 1 并立刻返回；判定在后台任务里异步进行，
                     判为"出现"则 k 重置为 1 并记录实测间隔
"""
from __future__ import annotations

import asyncio
import math
import random
import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

from . import core

PENDING_TTL = 600  # 请求-响应之间的临时记录最多保留多少秒


class StyleProbPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._rng = random.Random()
        self._states: dict[str, core.SessionState] = {}
        self._stats: dict | None = None
        self._index: list[str] | None = None
        self._pending: dict[int, core.RoundInfo] = {}
        self._tasks: set[asyncio.Task] = set()
        self._save_lock = asyncio.Lock()
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ 配置
    # 所有配置读取都经过清洗：WebUI 或手改配置文件可能给出 None、字符串、NaN、越界值，
    # 这里保证任何情况下都不抛异常，无效值回退到默认并在 /风格状态 里提示。
    def _flag(self, key: str, default: bool) -> bool:
        return core.parse_bool(self.config.get(key, default), default)

    def _text(self, key: str) -> str:
        v = self.config.get(key, "")
        return v.strip() if isinstance(v, str) else ""

    def _intensity(self) -> str:
        v = self.config.get("intensity", "mid")
        return v if isinstance(v, str) and (v in core.PRESETS or v == "custom") else "mid"

    def _params(self) -> tuple[float, float, int]:
        """低/中/高三档用内置预设；custom 档用 p0 / p_max / ramp_steps（经校验）。"""
        preset = core.PRESETS.get(self._intensity())
        if preset is not None:
            return core.normalize(*preset)
        params, warns = core.sanitize_params(
            self.config.get("p0", 0.05), self.config.get("p_max", 0.35),
            self.config.get("ramp_steps", 20))
        for w in warns:
            if w not in self._warned:  # 同一条提示只写一次日志
                self._warned.add(w)
                logger.warning(f"[style_prob] 配置问题：{w}")
        return params

    def _instruction(self) -> str:
        return self._text("style_instruction")

    def _inject_mode(self) -> str:
        v = self.config.get("inject_mode", "temp_part")
        return v if v in ("temp_part", "prompt") else "temp_part"

    def _timeout(self) -> int:
        v = core.to_int(self.config.get("judge_timeout", 30))
        return min(v, 300) if v is not None and v >= 1 else 30

    def _scope_sessions(self) -> set[str]:
        return set(core.clean_str_list(self.config.get("scope_sessions", [])))

    def _keywords(self) -> list[str]:
        return core.clean_str_list(self.config.get("keyword_hints", []))

    def _in_scope(self, umo: str) -> bool:
        if self.config.get("scope_mode", "all") != "specified":
            return True
        return umo in self._scope_sessions()

    def _config_warnings(self) -> list[str]:
        """当前配置里有问题的地方，给 /风格状态 展示。"""
        warns = []
        if self._intensity() == "custom":
            warns += core.sanitize_params(self.config.get("p0", 0.05), self.config.get("p_max", 0.35),
                                          self.config.get("ramp_steps", 20))[1]
        if not self._instruction():
            warns.append("风格指令为空，插件不会注入任何内容")
        t = core.to_int(self.config.get("judge_timeout", 30))
        if t is None or not 1 <= t <= 300:
            warns.append(f"判定超时 = {self.config.get('judge_timeout')!r} 无效（需要 1~300 的整数秒），"
                         f"已改用 {self._timeout()}")
        if self.config.get("scope_mode", "all") == "specified" and not self._scope_sessions():
            warns.append("生效范围是“仅指定会话”，但会话列表为空，当前不会对任何会话生效")
        return warns

    # ------------------------------------------------------------------ 持久化
    async def _get_state(self, umo: str) -> core.SessionState:
        st = self._states.get(umo)
        if st is None:
            raw = await self.get_kv_data(f"state:{umo}", None)
            st = core.SessionState.from_dict(raw)  # 数据损坏时回到 k=1
            self._states[umo] = st
        return st

    def _fingerprint(self) -> str:
        """当前生效的频率参数 + 风格指令。统计数据记录它，用来发现“改过设置”。"""
        return f"{self._params()}|{self._instruction()}"

    async def _get_stats(self) -> dict:
        if self._stats is None:
            raw = await self.get_kv_data("stats", None)
            if not isinstance(raw, dict):  # 数据损坏或不存在：从空统计开始
                raw = {}

            def num(key):  # 读取计数器；坏值按 0 处理
                v = core.to_int(raw.get(key, 0))
                return max(v, 0) if v is not None else 0

            ivs = raw.get("intervals", [])
            ivs = [i for i in (core.to_int(x) for x in ivs) if i is not None and i >= 1] \
                if isinstance(ivs, list) else []
            fp = raw.get("fp")
            self._stats = {
                "intervals": ivs,
                "injected": num("injected"),
                "messages": num("messages"),        # 计入统计的回复条数
                "judge_failed": num("judge_failed"),
                "inj_judged": num("inj_judged"),    # 被注入且已判定的轮数
                "inj_hits": num("inj_hits"),        # 其中风格确实出现的轮数
                "judged": num("judged"),
                "hits": num("hits"),
                "fp": fp if isinstance(fp, str) and fp else self._fingerprint(),
            }
        return self._stats

    async def _stats_stale(self) -> bool:
        """统计数据是否基于旧设置（改过档位/概率参数/风格指令后为 True）。"""
        return (await self._get_stats())["fp"] != self._fingerprint()

    async def _get_index(self) -> list[str]:
        """所有有过状态记录的会话，供“清空全部”使用。"""
        if self._index is None:
            raw = await self.get_kv_data("sessions", None)
            self._index = [s for s in raw if isinstance(s, str)] if isinstance(raw, list) else []
        return self._index

    async def _persist(self, umo: str | None = None):
        """串行写入，并在持锁后才取快照，保证后写入的一定是最新值。"""
        async with self._save_lock:
            if umo is not None and umo in self._states:
                index = await self._get_index()
                if umo not in index:
                    index.append(umo)
                    await self.put_kv_data("sessions", list(index))
                await self.put_kv_data(f"state:{umo}", self._states[umo].to_dict())
            if self._stats is not None:
                self._stats["intervals"] = self._stats["intervals"][-core.MAX_INTERVALS:]
                await self.put_kv_data("stats", dict(self._stats))

    # ------------------------------------------------------------------ 钩子：请求前
    @filter.on_llm_request()
    async def on_req(self, event: AstrMessageEvent, req: ProviderRequest):
        try:  # 插件自身的任何问题都不能影响用户的正常对话：出错就放行
            await self._on_req(event, req)
        except Exception as e:
            logger.error(f"[style_prob] 请求前处理出错，本轮不注入: {e!r}")

    async def _on_req(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._flag("enabled", True):
            return
        instruction = self._instruction()
        umo = event.unified_msg_origin
        if not instruction or not self._in_scope(umo):
            return
        key = id(event)
        if key in self._pending:  # 同一事件内的重复请求（如工具调用循环）不再抽样
            return
        self._prune_pending()

        st = await self._get_state(umo)
        p = core.prob_at(st.k, *self._params())
        injected = self._rng.random() < p
        if injected:
            # 只动本轮用户消息末尾；system_prompt、contexts 保持原样以保护缓存前缀
            injected = self._inject(req, instruction)
        self._pending[key] = core.RoundInfo(umo=umo, k=st.k, injected=injected)

    def _inject(self, req: ProviderRequest, instruction: str) -> bool:
        """prompt 模式：追加到 req.prompt（会随用户消息写入历史）。
        temp_part 模式：作为临时内容块追加（AstrBot>=4.24，不写入历史）；不可用时回退到 prompt。"""
        if self._inject_mode() == "temp_part":
            try:
                from astrbot.core.agent.message import TextPart
                req.extra_user_content_parts.append(
                    TextPart(text=f"\n{instruction}").mark_as_temp())
                return True
            except Exception as e:
                logger.warning(f"[style_prob] temp_part 模式不可用，回退到 prompt 模式: {e!r}")
        return core.inject_style(req, instruction)

    # ------------------------------------------------------------------ 钩子：回复后
    @filter.on_llm_response()
    async def on_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        try:
            await self._on_resp(event, resp)
        except Exception as e:
            logger.error(f"[style_prob] 回复后处理出错，本轮不计数: {e!r}")

    async def _on_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        key = id(event)
        info = self._pending.get(key)
        if info is None:
            return
        if getattr(resp, "tools_call_name", None):  # 工具调用的中间轮次，还不是最终回复
            return
        self._pending.pop(key, None)

        reply = (getattr(resp, "completion_text", "") or "").strip()
        if getattr(resp, "role", "assistant") == "err" or not reply:
            return  # 这一轮没有产生有效回复，不计数

        st = await self._get_state(info.umo)
        st.k += 1  # 先按"未出现"计数；判定为出现时再重置，判定失败也不会让 k 停滞
        stats = await self._get_stats()
        stats["messages"] += 1
        if info.injected:
            stats["injected"] += 1

        do_judge = info.injected or not self._flag("judge_only_when_injected", True)
        user_text = ""
        if do_judge and self._flag("judge_include_user", False):
            user_text = (event.get_message_str() or "").strip()

        # 不 await：判定与落盘都放到后台任务，立即返回，不拖慢回复发送
        self._spawn(self._finish_round(info, reply, user_text, do_judge))

    # ------------------------------------------------------------------ 后台任务
    def _spawn(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _prune_pending(self):
        now = time.time()
        for k in [k for k, v in self._pending.items() if now - v.ts > PENDING_TTL]:
            self._pending.pop(k, None)

    async def _finish_round(self, info: core.RoundInfo, reply: str,
                            user_text: str, do_judge: bool):
        try:
            await self._persist(info.umo)
            if not do_judge:
                return
            hit = await self._judge(info.umo, reply, user_text)
            stats = await self._get_stats()
            if hit is None:
                stats["judge_failed"] += 1
                await self._persist(info.umo)
                return
            stats["judged"] += 1
            if info.injected:
                stats["inj_judged"] += 1
                stats["inj_hits"] += int(hit)
            if hit:
                stats["hits"] += 1
                stats["intervals"].append(info.k)  # 间隔 T = 触发时的 k
                (await self._get_state(info.umo)).k = 1
            await self._persist(info.umo)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # 后台任务的异常不能悄悄吞掉
            logger.error(f"[style_prob] 判定任务失败: {e!r}")

    # ------------------------------------------------------------------ 判定
    async def _judge(self, umo: str, reply: str, user_text: str, *,
                     instruction: str | None = None, template: str | None = None,
                     use_keywords: bool = True) -> bool | None:
        """True=风格出现，False=没出现，None=无法判定（不改变状态）。
        instruction / template / use_keywords 仅供自测覆盖，正常流程用配置。"""
        if use_keywords and core.keyword_hit(reply, self._keywords()):
            return True  # 关键词只用于"确定是"；没命中不代表"否"，继续交给模型

        prompt = core.build_judge_prompt(
            self._text("judge_prompt") if template is None else template,
            self._instruction() if instruction is None else instruction,
            reply, user_text,
        )
        try:
            pid = self._text("judge_provider")
            if not pid:
                pid = await self.context.get_current_chat_provider_id(umo=umo)
            text = await asyncio.wait_for(self._call_llm(pid, prompt), timeout=self._timeout())
        except Exception as e:
            logger.warning(f"[style_prob] 判定模型调用失败，本轮不更新: {e!r}")
            return None

        verdict = core.parse_yes_no(text)
        if verdict is None:
            logger.warning(f"[style_prob] 判定结果无法解析为是/否: {text!r}")
        return verdict

    async def _call_llm(self, provider_id: str, prompt: str) -> str:
        # 单独一次调用，不带历史对话
        if hasattr(self.context, "llm_generate"):
            r = await self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt)
        else:  # 老版本回退
            prov = self.context.get_provider_by_id(provider_id)
            r = await prov.text_chat(prompt=prompt)
        return (r.completion_text or "").strip()

    # ------------------------------------------------------------------ 指令
    @filter.command("风格概率")
    async def cmd_prob(self, event: AstrMessageEvent):
        """显示理论 E[T]、每 100 条预期触发次数，以及实测统计"""
        p0, pm, n = self._params()
        et = core.expected_interval(p0, pm, n)
        per100 = core.expected_triggers(100, p0, pm, n)
        stats = await self._get_stats()
        intervals = stats["intervals"]

        lines = [
            f"强度档位：{core.PRESET_LABELS[self._intensity()]}    p0={p0:g}  p_max={pm:g}  n={n}",
            "平均触发间隔 E[T] = " + ("∞（p_max=0，永不触发）" if et == float("inf") else f"{et:.2f} 条"),
            f"每 100 条消息预期被注入约 {per100:.2f} 次",
            "（这是“被注入”的理论值，模型不一定照做，实际出现频率会更低）",
        ]
        if intervals:
            mean = sum(intervals) / len(intervals)
            lines.append(f"实测：{len(intervals)} 个间隔，平均 {mean:.2f} 条")
        else:
            lines.append("实测：暂无数据")
        if await self._stats_stale():
            lines.append("提示：设置已更改，实测数据来自旧设置，可用 /风格清空 清除。")
        lines += [f"⚠ {w}" for w in self._config_warnings()]
        yield event.plain_result("\n".join(lines))

    @filter.command("风格统计")
    async def cmd_stats(self, event: AstrMessageEvent):
        """对比预期与实际的触发次数，并给出样本量参考"""
        yield event.plain_result(await self._render_stats())

    async def _render_stats(self) -> str:
        params = self._params()
        stats = await self._get_stats()
        mean, _ = core.interval_moments(*params)
        msgs, hits = stats["messages"], stats["hits"]
        lines = [f"【统计】强度档位：{core.PRESET_LABELS[self._intensity()]}（自上次清空起）"]
        lines.append(
            f"样本：回复 {msgs} 条，注入 {stats['injected']} 次，判定 {stats['judged']} 次，"
            f"判为出现 {hits} 次，判定失败 {stats['judge_failed']} 次")
        if await self._stats_stale():
            # 数据混有新旧设置，拿当前设置算“预期值”会得出误导性的结论，所以不做对比和检验
            lines.append("设置已更改（档位 / 概率参数 / 风格指令），这批数据混有新旧设置，"
                         "不做预期对比和检验。请先 /风格清空，再重新累计。")
            return "\n".join(lines)
        if math.isinf(mean):
            lines.append("当前设置下概率为 0，不会触发。")
            return "\n".join(lines)

        t = core.count_test(msgs, hits, *params)
        if msgs == 0:
            lines.append("还没有数据。")
        else:
            dev = f"{t['deviation'] * 100:+.0f}%" if t["deviation"] is not None else "-"
            lines.append(f"触发次数：预期 {t['expected']:.1f} 次（{msgs} ÷ {mean:.2f}），"
                         f"实际 {hits} 次（{dev}）")
            p = t["p_value"]
            ptxt = "<0.001" if p < 0.001 else f"{p:.3f}"
            if not t["reliable"]:
                verdict = (f"预期触发不足 {core.MIN_EXPECTED_TRIGGERS} 次，样本太少，"
                           "差异大小暂不能当结论")
            elif p >= 0.05:
                verdict = "与预期无显著差异"
            elif t["actual"] < t["expected"]:
                verdict = "显著低于预期：模型没有完全遵循指令，或判定偏严"
            else:
                verdict = "显著高于预期：可能是模型自发出现了该风格（关闭“仅在注入时判定”时会如此），或判定偏松"
            lines.append(f"检验：z = {t['z']:+.2f}，p = {ptxt} → {verdict}")

        if stats["inj_judged"]:
            lo, hi = core.wilson_interval(stats["inj_hits"], stats["inj_judged"])
            lines.append(
                f"遵循率（被注入后风格确实出现）：{stats['inj_hits'] / stats['inj_judged'] * 100:.0f}%"
                f"（95% 区间 {lo * 100:.0f}%~{hi * 100:.0f}%，样本 {stats['inj_judged']} 次）")
        if stats["intervals"]:
            mi = sum(stats["intervals"]) / len(stats["intervals"])
            lines.append(f"平均触发间隔：实测 {mi:.2f} 条（{len(stats['intervals'])} 个），理论 {mean:.2f} 条")

        lines.append("样本量参考（95% 置信、80% 把握看出偏差；累计触发次数 / 约需消息条数 / 进度）：")
        for dev in (0.3, 0.2, 0.1):
            need = core.required_triggers(*params, dev)
            mark = "已达到" if hits >= need else f"{hits}/{need}"
            lines.append(f"  看出 ±{dev * 100:.0f}% 的偏差：约 {need} 次 / 约 {need * mean:.0f} 条 / {mark}")
        return "\n".join(lines)

    @filter.command("风格状态")
    async def cmd_status(self, event: AstrMessageEvent):
        """查看当前会话的 k 与当前概率"""
        umo = event.unified_msg_origin
        st = await self._get_state(umo)
        stats = await self._get_stats()
        p = core.prob_at(st.k, *self._params())
        scope = "是" if self._in_scope(umo) else "否（不在生效范围内）"
        text = (
            f"当前会话：距上次触发已过 {st.k - 1} 条消息，下一条为第 k = {st.k} 条，注入概率 {p:.3f}\n"
            f"强度档位：{core.PRESET_LABELS[self._intensity()]}；本会话在生效范围内：{scope}\n"
            f"累计（全部会话）：注入 {stats['injected']} 次，判定 {stats['judged']} 次，"
            f"判为出现 {stats['hits']} 次，已记录 {len(stats['intervals'])} 个触发间隔"
        )
        if await self._stats_stale():
            text += "\n提示：频率或风格指令的设置已更改，上面的累计数据来自旧设置，可用 /风格清空 清除。"
        text += "".join(f"\n⚠ {w}" for w in self._config_warnings())
        yield event.plain_result(text)

    @filter.command("风格清空")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cmd_clear(self, event: AstrMessageEvent):
        """清空所有会话的计数和全部统计数据（管理员）。修改档位或频率设置后可用"""
        for t in list(self._tasks):  # 先停掉后台判定，避免它把旧数据写回来
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        async with self._save_lock:
            index = await self._get_index()
            for umo in index:
                await self.delete_kv_data(f"state:{umo}")
            await self.delete_kv_data("stats")
            await self.delete_kv_data("sessions")
            count = len(index)
            self._states.clear()
            self._pending.clear()
            self._index = []
            self._stats = None
        yield event.plain_result(f"已清空：{count} 个会话的计数与全部统计数据，所有会话从 k = 1 重新开始。")

    @filter.command("风格重置")
    async def cmd_reset(self, event: AstrMessageEvent):
        """把当前会话的计数重置为 k=1（概率回到 p_0）"""
        umo = event.unified_msg_origin
        (await self._get_state(umo)).k = 1
        await self._persist(umo)
        yield event.plain_result(f"已重置：k = 1，当前概率 = {core.prob_at(1, *self._params()):.3f}")

    # 自测专用：指令和测试消息都写死，不读配置，保证每次测的是同一件事。
    # 风格选“句末加喵~”是因为特征明确，模型有没有照做一眼就能看出来。
    SELFTEST_INSTRUCTION = "请在回复的每一句话末尾都加上“喵~”。"
    SELFTEST_MESSAGES = [
        "今天中午吃什么好呢？",
        "给我讲个冷笑话吧。",
        "简单介绍一下你自己。",
        "明天要考试了，有点紧张。",
        "你觉得周末适合做什么？",
    ]

    @filter.command("风格自测")
    async def cmd_selftest(self, event: AstrMessageEvent, n: int = 3):
        """实测：用写死的“句末加喵~”指令 100% 注入，让模型回答 n 次，并对照不注入的结果，用判定模型检查"""
        instruction = self.SELFTEST_INSTRUCTION
        n = core.to_int(n)                 # 非整数、负数、超大数都规整到 1~10
        n = 3 if n is None else min(max(n, 1), 10)
        umo = event.unified_msg_origin
        try:
            pid = await self.context.get_current_chat_provider_id(umo=umo)
        except Exception as e:
            yield event.plain_result(f"取不到当前会话的聊天模型：{e}")
            return
        yield event.plain_result(
            f"开始自测：指令固定为“{instruction}”。{n} 条消息，每条各问一次“注入”和“不注入”，"
            f"共 {2 * n} 次生成 + {2 * n} 次判定，请稍等……")

        async def ask(msg: str, inject: bool):
            prompt = f"{msg}\n\n{instruction}" if inject else msg  # 与 prompt 模式的拼接格式一致
            try:
                reply = await asyncio.wait_for(
                    self._call_llm(pid, prompt),
                    timeout=self._timeout() * 2)
            except Exception as e:
                return None, None, f"生成失败：{e!r}"
            # 用内置判定模板 + 写死的指令，不用关键词，只测"模型听不听话"和"判定模型准不准"
            verdict = await self._judge(umo, reply, "", instruction=instruction,
                                        template="", use_keywords=False)
            return reply, verdict, ""

        def mark(v):
            return "出现" if v is True else "未出现" if v is False else "判定失败"

        lines, hits = [], {True: 0, False: 0}
        for i in range(n):
            msg = self.SELFTEST_MESSAGES[i % len(self.SELFTEST_MESSAGES)]
            lines.append(f"【{i + 1}】用户：{msg}")
            for inject in (True, False):
                reply, verdict, err = await ask(msg, inject)
                tag = "注入" if inject else "对照"
                if err:
                    lines.append(f"  {tag}：{err}")
                    continue
                hits[inject] += verdict is True
                short = reply.replace("\n", " ")
                short = short[:60] + ("…" if len(short) > 60 else "")
                lines.append(f"  {tag}：{mark(verdict)} | {short}")
        lines.append(f"结果：注入组 {hits[True]}/{n} 次出现风格，对照组 {hits[False]}/{n} 次。")
        lines.append("（测试指令固定为：" + instruction + "；用当前会话的聊天模型直接生成，不带人设和历史，"
                     "判定用内置模板。检验的是模型听不听话、判定模型准不准；不会改动当前会话的 k。）")
        yield event.plain_result("\n".join(lines))

    # ------------------------------------------------------------------ 其它
    async def terminate(self):
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
