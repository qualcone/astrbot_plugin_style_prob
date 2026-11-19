"""用桩模块替换 astrbot，验证插件钩子的行为：缓存前缀不变、判定不阻塞回复。"""
import asyncio
import copy
import logging
import sys
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _install_astrbot_stubs():
    def mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    astrbot, api = mod("astrbot"), mod("astrbot.api")
    event, star = mod("astrbot.api.event"), mod("astrbot.api.star")
    provider, comp = mod("astrbot.api.provider"), mod("astrbot.api.message_components")
    astrbot.api = api
    for sub, name in ((event, "event"), (star, "star"), (provider, "provider"),
                      (comp, "message_components")):
        setattr(api, name, sub)

    api.AstrBotConfig = dict
    api.logger = logging.getLogger("stub")

    ident = lambda *a, **k: (lambda f: f)  # noqa: E731
    event.filter = SimpleNamespace(on_llm_request=ident, on_llm_response=ident, command=ident,
                                   permission_type=ident, PermissionType=SimpleNamespace(ADMIN=1))
    event.AstrMessageEvent = object

    class Star:
        def __init__(self, context):
            self.context = context
            self._kv = {}

        async def put_kv_data(self, k, v):
            self._kv[k] = copy.deepcopy(v)

        async def get_kv_data(self, k, default):
            return copy.deepcopy(self._kv.get(k, default))

        async def delete_kv_data(self, k):
            self._kv.pop(k, None)

    star.Star = Star
    star.Context = object
    provider.ProviderRequest = object
    provider.LLMResponse = object
    comp.Plain = comp.Image = object

    # astrbot.core.agent.message.TextPart（temp_part 模式用）
    for n in ("astrbot.core", "astrbot.core.agent"):
        mod(n)
    msg = mod("astrbot.core.agent.message")

    class TextPart:
        def __init__(self, text):
            self.text, self.temp = text, False

        def mark_as_temp(self):
            self.temp = True
            return self

    msg.TextPart = TextPart


_install_astrbot_stubs()
from astrbot_plugin_style_prob import core  # noqa: E402
from astrbot_plugin_style_prob.main import StyleProbPlugin  # noqa: E402


class FakeContext:
    def __init__(self, answer="是", delay=0.4):
        self.answer, self.delay, self.calls = answer, delay, []

    async def get_current_chat_provider_id(self, umo):
        return "chat-default"

    async def llm_generate(self, *, chat_provider_id, prompt, **kw):
        self.calls.append({"provider": chat_provider_id, "prompt": prompt, "kw": kw})
        await asyncio.sleep(self.delay)
        return SimpleNamespace(completion_text=self.answer)


def make_plugin(ctx, **cfg):
    base = {"enabled": True, "style_instruction": "请撒娇", "p0": 1.0, "p_max": 1.0,
            "intensity": "custom",
            "ramp_steps": 5, "inject_mode": "prompt", "judge_provider": "cheap", "judge_prompt": "",
            "judge_include_user": False, "keyword_hints": [],
            "judge_only_when_injected": True, "judge_timeout": 5,
            "scope_sessions": [], "scope_personas": []}
    base.update(cfg)
    plugin = StyleProbPlugin(ctx, base)
    plugin.FLUSH_DELAY = 0  # 测试里不等待合并写入的延迟
    return plugin


def make_event(umo="u1", msg="你好"):
    return SimpleNamespace(unified_msg_origin=umo, get_message_str=lambda: msg,
                           plain_result=lambda t: t)


def make_req():
    return SimpleNamespace(
        prompt="你好", system_prompt="SYSTEM",
        contexts=[{"role": "user", "content": "旧消息"}, {"role": "assistant", "content": "旧回复"}],
        image_urls=[])


def make_resp(text="喵～", **kw):
    return SimpleNamespace(role="assistant", completion_text=text, tools_call_name=[], **kw)


class PluginFlow(unittest.IsolatedAsyncioTestCase):
    async def test_cache_prefix_untouched(self):
        plugin = make_plugin(FakeContext())
        ev, req = make_event(), make_req()
        before = copy.deepcopy(req)
        await plugin.on_req(ev, req)
        self.assertEqual(req.system_prompt, before.system_prompt)      # system prompt 不变
        self.assertEqual(req.contexts, before.contexts)                # 历史消息不变
        self.assertEqual(req.prompt, before.prompt + "\n\n请撒娇")      # 只在末尾追加
        await plugin.terminate()

    async def test_temp_part_mode_leaves_prompt_alone(self):
        plugin = make_plugin(FakeContext(), inject_mode="temp_part")
        req = make_req()
        req.extra_user_content_parts = []
        before = copy.deepcopy(req.__dict__)
        await plugin.on_req(make_event(), req)
        self.assertEqual(req.prompt, before["prompt"])
        self.assertEqual(req.system_prompt, before["system_prompt"])
        self.assertEqual(req.contexts, before["contexts"])
        self.assertEqual(len(req.extra_user_content_parts), 1)
        part = req.extra_user_content_parts[0]
        self.assertTrue(part.temp)
        self.assertIn("请撒娇", part.text)

    async def test_default_mode_is_temp_part(self):
        cfg_plugin = make_plugin(FakeContext())
        del cfg_plugin.config["inject_mode"]           # 未配置时取默认
        req = make_req()
        req.extra_user_content_parts = []
        await cfg_plugin.on_req(make_event(), req)
        self.assertEqual(req.prompt, "你好")
        self.assertEqual(len(req.extra_user_content_parts), 1)

    async def test_temp_part_falls_back_to_prompt(self):
        plugin = make_plugin(FakeContext(), inject_mode="temp_part")
        req = make_req()                                # 没有 extra_user_content_parts 属性
        await plugin.on_req(make_event(), req)
        self.assertTrue(req.prompt.endswith("请撒娇"))

    async def test_selftest_command(self):
        class Smart(FakeContext):
            """模拟一个会听话的模型：带指令就撒娇；判定提示里看到“喵”就答是。"""
            async def llm_generate(self, *, chat_provider_id, prompt, **kw):
                if "只回答一个字" in prompt:            # 判定调用
                    # 只看机器人回复那一段，避免被风格指令本身里的字干扰
                    reply = prompt.split("机器人的回复：", 1)[1]
                    return SimpleNamespace(completion_text="是" if "喵" in reply else "否")
                if StyleProbPlugin.SELFTEST_INSTRUCTION in prompt:
                    return SimpleNamespace(completion_text="好呀喵~")
                return SimpleNamespace(completion_text="好的。")

        # 用户配置的指令/判定模板/关键词都不应影响自测
        plugin = make_plugin(Smart(delay=0), style_instruction="完全无关的指令",
                             judge_prompt="自定义模板：{reply}", keyword_hints=["好的"])
        out = [x async for x in plugin.cmd_selftest(make_event(), 3)]
        text = out[-1]
        self.assertIn(StyleProbPlugin.SELFTEST_INSTRUCTION, out[0])
        self.assertIn("注入组 3/3", text)
        self.assertIn("对照组 0/3", text)
        self.assertEqual(plugin._states.get("u1"), None)   # 自测不改会话状态

    async def test_intensity_presets(self):
        for tier, (p0, pm, n) in core.PRESETS.items():
            plugin = make_plugin(FakeContext(), intensity=tier, p0=0.9, p_max=0.9, ramp_steps=1)
            self.assertEqual(plugin._params(), (p0, pm, n), tier)   # 预设覆盖自定义值
        plugin = make_plugin(FakeContext(), intensity="custom", p0=0.1, p_max=0.2, ramp_steps=3)
        self.assertEqual(plugin._params(), (0.1, 0.2, 3))
        plugin = make_plugin(FakeContext(), intensity="bogus")
        self.assertEqual(plugin._params(), core.normalize(*core.PRESETS["mid"]))

    async def _run_round(self, plugin, umo, answer_hit):
        ev = make_event(umo)
        await plugin.on_req(ev, make_req())
        await plugin.on_resp(ev, make_resp())
        await plugin._wait_tasks()

    async def test_state_survives_restart(self):
        p1 = make_plugin(FakeContext(answer="否", delay=0.01))
        for _ in range(3):
            await self._run_round(p1, "u1", False)
        self.assertEqual(p1._states["u1"].k, 4)
        p2 = make_plugin(FakeContext(delay=0.01))      # 模拟重启：新实例，共用同一份 KV 存储
        p2._kv = p1._kv
        status = [x async for x in p2.cmd_status(make_event("u1"))][0]
        self.assertEqual(p2._states["u1"].k, 4)
        self.assertIn("k = 4", status)
        self.assertIn("已过 3 条", status)

    async def test_clear_all_and_stale_hint(self):
        p = make_plugin(FakeContext(answer="否", delay=0.01))
        for umo in ("u1", "u2"):
            await self._run_round(p, umo, False)
        self.assertEqual(p._kv["sessions"], ["u1", "u2"])
        self.assertFalse(await p._stats_stale())
        p.config["p0"] = 0.5                           # 改了设置（p_max 仍为 1.0）
        self.assertTrue(await p._stats_stale())
        self.assertIn("可用 /风格清空", [x async for x in p.cmd_status(make_event("u1"))][0])

        out = [x async for x in p.cmd_clear(make_event("u1"))][0]
        self.assertIn("2 个会话", out)
        self.assertEqual([k for k in p._kv if k.startswith(("state:", "stats", "sessions"))], [])
        self.assertEqual(p._states, {})
        self.assertFalse(await p._stats_stale())       # 清空后按新设置重新计
        st = await p._get_state("u1")
        self.assertEqual(st.k, 1)

    async def test_stats_report(self):
        p = make_plugin(FakeContext(answer="是", delay=0.01))   # p0=p_max=1：E[T]=1，每条都注入且都出现
        for _ in range(5):
            await self._run_round(p, "u1", True)
        s = p._stats
        self.assertEqual((s["messages"], s["injected"], s["judged"], s["hits"]), (5, 5, 5, 5))
        self.assertEqual((s["inj_judged"], s["inj_hits"]), (5, 5))
        text = [x async for x in p.cmd_stats(make_event())][0]
        self.assertIn("回复 5 条", text)
        self.assertIn("预期 5.0 次", text)
        self.assertIn("实际 5 次", text)
        self.assertIn("样本太少", text)                 # 预期触发 < 30 次
        self.assertIn("遵循率", text)
        self.assertIn("100%", text)
        self.assertIn("样本量参考", text)

    async def test_stats_significantly_low(self):
        p = make_plugin(FakeContext(answer="否", delay=0.0), intensity="custom",
                        p0=0.5, p_max=0.5)              # E[T]=2：100 条应约 50 次，但判定全是“否”
        for _ in range(100):
            await self._run_round(p, "u1", False)
        text = [x async for x in p.cmd_stats(make_event())][0]
        self.assertIn("实际 0 次", text)
        self.assertIn("显著低于预期", text)

    async def test_judge_failure_is_counted(self):
        class Boom(FakeContext):
            async def llm_generate(self, **kw):
                raise RuntimeError("down")
        p = make_plugin(Boom())
        await self._run_round(p, "u1", False)
        self.assertEqual(p._stats["judge_failed"], 1)
        self.assertEqual(p._stats["judged"], 0)

    async def test_reset_only_current_session(self):
        p = make_plugin(FakeContext(answer="否", delay=0.01))
        for umo in ("u1", "u2"):
            await self._run_round(p, umo, False)
        [x async for x in p.cmd_reset(make_event("u1"))]
        self.assertEqual(p._states["u1"].k, 1)
        self.assertEqual(p._states["u2"].k, 2)

    async def test_no_injection_when_probability_zero(self):
        plugin = make_plugin(FakeContext(), p0=0.0, p_max=0.0)
        req = make_req()
        await plugin.on_req(make_event(), req)
        self.assertEqual(req.prompt, "你好")

    async def test_resolve_persona_fallback_chain(self):
        class PM:
            default_persona = "default"

        class CM:
            def __init__(self, pid):
                self.pid = pid

            async def get_curr_conversation_id(self, umo):
                return "c1"

            async def get_conversation(self, umo, cid):
                return SimpleNamespace(persona_id=self.pid)

        ctx = FakeContext()
        ctx.persona_manager = PM()
        ctx.conversation_manager = CM("cat")
        plugin = make_plugin(ctx, scope_personas=["cat"])
        self.assertEqual(await plugin._resolve_persona(make_event()), "cat")
        ctx.conversation_manager = CM(None)   # 会话未指定人设：回落到默认人设
        self.assertEqual(await plugin._resolve_persona(make_event()), "default")
        self.assertIsNone(await make_plugin(FakeContext())._resolve_persona(make_event()))

    async def test_judge_does_not_block_and_hit_resets(self):
        ctx = FakeContext(answer="是", delay=0.4)
        plugin = make_plugin(ctx)
        plugin._states["u1"] = core.SessionState(k=6)
        ev = make_event()
        await plugin.on_req(ev, make_req())

        t0 = time.perf_counter()
        await plugin.on_resp(ev, make_resp())
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.1, "on_llm_response 不应等待判定模型")  # 判定要 0.4s
        self.assertEqual(len(plugin._tasks), 1)                         # 判定在后台
        self.assertEqual(plugin._states["u1"].k, 7)                     # 先按"未出现"计数

        await plugin._wait_tasks()
        self.assertEqual(plugin._states["u1"].k, 1)                     # 判为是 -> 重置
        self.assertEqual(plugin._stats["intervals"], [6])               # 记录间隔 T=6
        self.assertEqual(plugin._kv["state:u1"]["k"], 1)                # 已持久化
        # 判定调用不带历史：只有 prompt，没有 contexts
        self.assertNotIn("contexts", ctx.calls[0]["kw"])
        self.assertEqual(ctx.calls[0]["provider"], "cheap")

    async def test_miss_keeps_counting(self):
        plugin = make_plugin(FakeContext(answer="否", delay=0.01))
        ev = make_event()
        await plugin.on_req(ev, make_req())
        await plugin.on_resp(ev, make_resp())
        await plugin._wait_tasks()
        self.assertEqual(plugin._states["u1"].k, 2)
        self.assertEqual(plugin._stats["intervals"], [])

    async def test_judge_only_when_injected(self):
        ctx = FakeContext(delay=0.01)
        plugin = make_plugin(ctx, p0=0.0, p_max=0.0)   # 永不注入
        ev = make_event()
        await plugin.on_req(ev, make_req())
        await plugin.on_resp(ev, make_resp())
        await plugin._wait_tasks()
        self.assertEqual(ctx.calls, [])                 # 没注入 -> 不判定
        self.assertEqual(plugin._states["u1"].k, 2)     # 但 k 仍然 +1

    async def test_keyword_fast_path_skips_model(self):
        ctx = FakeContext(delay=0.01)
        plugin = make_plugin(ctx, keyword_hints=["喵"])
        ev = make_event()
        await plugin.on_req(ev, make_req())
        await plugin.on_resp(ev, make_resp("喵～好的"))
        await plugin._wait_tasks()
        self.assertEqual(ctx.calls, [])
        self.assertEqual(plugin._states["u1"].k, 1)

    async def test_tool_call_round_is_ignored(self):
        plugin = make_plugin(FakeContext(delay=0.01))
        ev = make_event()
        await plugin.on_req(ev, make_req())
        mid = SimpleNamespace(role="assistant", completion_text="", tools_call_name=["search"])
        await plugin.on_resp(ev, mid)
        self.assertEqual(plugin._states["u1"].k, 1)     # 中间轮次不计数
        await plugin.on_resp(ev, make_resp())            # 最终回复才计数
        await plugin._wait_tasks()

    async def test_judge_failure_does_not_crash_or_reset(self):
        class Boom(FakeContext):
            async def llm_generate(self, **kw):
                raise RuntimeError("provider down")

        plugin = make_plugin(Boom())
        ev = make_event()
        await plugin.on_req(ev, make_req())
        await plugin.on_resp(ev, make_resp())
        await plugin._wait_tasks()
        self.assertEqual(plugin._states["u1"].k, 2)


if __name__ == "__main__":
    unittest.main()
