"""边界与格式检查：所有数字输入、坏配置、坏数据、改设置不清空。"""
import asyncio
import json
import math
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from test_plugin_flow import (FakeContext, make_event, make_plugin, make_req,  # noqa: E402
                              make_resp)
from astrbot_plugin_style_prob import core  # noqa: E402

NAN, INF = float("nan"), float("inf")
PLUGIN_DIR = Path(__file__).resolve().parents[1]


class Conversions(unittest.TestCase):
    def test_to_float(self):
        for v, want in [(0.3, 0.3), ("0.3", 0.3), (" 5 ", 5.0), (1, 1.0), (0, 0.0)]:
            self.assertEqual(core.to_float(v), want, v)
        for v in (None, True, False, "abc", "", "35%", NAN, INF, -INF, "nan", "inf", "1e999", [], {}):
            self.assertIsNone(core.to_float(v), repr(v))

    def test_to_int(self):
        for v, want in [(20, 20), (20.0, 20), ("20", 20), (-3, -3), ("1e2", 100)]:
            self.assertEqual(core.to_int(v), want, v)
        for v in (20.5, "20.5", None, True, "abc", NAN, INF, []):
            self.assertIsNone(core.to_int(v), repr(v))

    def test_parse_bool(self):
        for v in (True, 1, "true", "TRUE", " yes ", "on", "1", "是"):
            self.assertTrue(core.parse_bool(v, False), repr(v))
        for v in (False, 0, "false", "False", "no", "off", "0", "", "否"):
            self.assertFalse(core.parse_bool(v, True), repr(v))
        self.assertTrue(core.parse_bool(None, True))
        self.assertTrue(core.parse_bool("maybe", True))     # 无法识别 -> 默认值
        self.assertFalse(core.parse_bool([], False))

    def test_clean_str_list(self):
        self.assertEqual(core.clean_str_list(None), [])
        self.assertEqual(core.clean_str_list(5), [])
        self.assertEqual(core.clean_str_list({"a": 1}), [])
        self.assertEqual(core.clean_str_list("喵\n 汪 \n\n"), ["喵", "汪"])        # 整串配置按行拆分
        self.assertEqual(core.clean_str_list([1, "x", None, True, {"a": 1}, ["z"], " y ", ""]),
                         ["1", "x", "y"])

    def test_session_state_garbage(self):
        for raw in (None, [], "x", 5, {}, {"k": None}, {"k": "bad"}, {"k": 2.5}, {"k": NAN}):
            self.assertEqual(core.SessionState.from_dict(raw).k, 1, repr(raw))
        self.assertEqual(core.SessionState.from_dict({"k": "7"}).k, 7)
        self.assertEqual(core.SessionState.from_dict({"k": -5}).k, 1)
        self.assertEqual(core.SessionState.from_dict({"k": 0}).k, 1)


class SanitizeParams(unittest.TestCase):
    def test_valid_values_pass_unchanged(self):
        for args in [(0.0, 0.0, 0), (1.0, 1.0, 0), (0.05, 0.35, 20), (0, 1, core.MAX_RAMP_STEPS),
                     ("0.1", "0.5", "10"), (0.2, 0.2, 5.0)]:
            params, warns = core.sanitize_params(*args)
            self.assertEqual(warns, [], args)
            self.assertEqual(params, (float(args[0]), float(args[1]), int(float(args[2]))))

    def test_invalid_probabilities_fall_back_with_warning(self):
        bad = [None, "abc", "", "35%", NAN, INF, -INF, -0.01, 1.0001, 35, 100, True, [], {}]
        for v in bad:
            (p0, pm, n), warns = core.sanitize_params(v, 0.5, 10)
            self.assertEqual(p0, core.DEFAULT_CUSTOM[0], repr(v))
            self.assertTrue(any("p0" in w for w in warns), repr(v))
            (p0, pm, n), warns = core.sanitize_params(0.01, v, 10)
            self.assertEqual(pm, core.DEFAULT_CUSTOM[1], repr(v))
            self.assertTrue(any("p_max" in w for w in warns), repr(v))

    def test_invalid_steps_fall_back_with_warning(self):
        for v in (None, "abc", -1, 20.5, NAN, INF, core.MAX_RAMP_STEPS + 1, 10 ** 9, True, []):
            (_, _, n), warns = core.sanitize_params(0.05, 0.35, v)
            self.assertEqual(n, core.DEFAULT_CUSTOM[2], repr(v))
            self.assertTrue(any("上升步数" in w for w in warns), repr(v))

    def test_pmax_below_p0_is_raised_to_p0(self):
        (p0, pm, n), warns = core.sanitize_params(0.5, 0.2, 10)
        self.assertEqual((p0, pm), (0.5, 0.5))
        self.assertEqual(len(warns), 1)

    def test_normalize_never_raises(self):
        garbage = [None, "abc", NAN, INF, -1, 99, True, [], {}, 10 ** 12]
        for a in garbage:
            for b in garbage:
                for c in garbage:
                    p0, pm, n = core.normalize(a, b, c)
                    self.assertTrue(0 <= p0 <= pm <= 1 and 0 <= n <= core.MAX_RAMP_STEPS, (a, b, c))
                    self.assertTrue(0 <= core.prob_at(1, a, b, c) <= 1)

    def test_prob_at_handles_nonpositive_k(self):
        for k in (0, -5, 1):
            self.assertAlmostEqual(core.prob_at(k, 0.1, 0.5, 10), 0.1)


class Extremes(unittest.TestCase):
    def test_huge_steps_does_not_hang(self):
        t0 = time.perf_counter()
        for n in (10 ** 9, core.MAX_RAMP_STEPS):
            core.interval_moments(0.05, 0.35, n)
            core.expected_interval(0.05, 0.35, n)
            core.required_triggers(0.05, 0.35, n, 0.2)
        self.assertLess(time.perf_counter() - t0, 3)

    def test_tiny_pmax_is_finite_and_fast(self):
        t0 = time.perf_counter()
        mean, sd = core.interval_moments(0.0, 1e-9, 10)
        self.assertTrue(math.isfinite(mean) and mean > 1e8)
        self.assertTrue(math.isfinite(sd))
        self.assertLess(time.perf_counter() - t0, 1)

    def test_nan_inf_params_do_not_hang(self):
        t0 = time.perf_counter()
        for bad in (NAN, INF, -INF):
            core.interval_moments(bad, bad, bad)
            core.expected_interval(bad, bad, bad)
        self.assertLess(time.perf_counter() - t0, 1)

    def test_degenerate_distributions(self):
        self.assertEqual(core.interval_moments(1.0, 1.0, 0), (1.0, 0.0))
        self.assertEqual(core.interval_moments(1.0, 1.0, 7), (1.0, 0.0))
        m, s = core.interval_moments(0.0, 0.0, 5)
        self.assertTrue(math.isinf(m) and math.isinf(s))
        self.assertEqual(core.required_triggers(1.0, 1.0, 5, 0.2), 0)    # 没有波动，不需要样本
        self.assertEqual(core.required_triggers(0.0, 0.0, 5, 0.2), 0)
        self.assertEqual(core.required_triggers(0.05, 0.35, 20, 0), 0)   # 偏差 0 无意义，不除零

    def test_count_test_zero_variance(self):
        same = core.count_test(10, 10, 1.0, 1.0, 0)
        self.assertEqual(same["z"], 0.0)
        off = core.count_test(10, 7, 1.0, 1.0, 0)        # 每条必触发，却只有 7 次：不可能是随机波动
        self.assertEqual(off["z"], -math.inf)
        self.assertEqual(off["p_value"], 0.0)
        self.assertEqual(core.count_test(10, 12, 1.0, 1.0, 0)["z"], math.inf)

    def test_count_test_degenerate_inputs(self):
        self.assertIsNone(core.count_test(0, 0, 0.05, 0.35, 20)["z"])
        self.assertIsNone(core.count_test(-5, 0, 0.05, 0.35, 20)["z"])
        self.assertIsNone(core.count_test(100, 0, 0.0, 0.0, 5)["z"])
        t = core.count_test(10 ** 9, 0, 0.05, 0.35, 20)
        self.assertEqual(t["p_value"], 0.0)

    def test_wilson_edges(self):
        for s, t in [(0, 1), (1, 1), (0, 100), (100, 100), (0, 0)]:
            lo, hi = core.wilson_interval(s, t)
            self.assertTrue(0.0 <= lo <= hi <= 1.0, (s, t))

    def test_build_judge_prompt_non_string_template(self):
        for tpl in (None, 5, [], {}):
            self.assertIn("回复", core.build_judge_prompt(tpl, "指令", "回复"))


class Schema(unittest.TestCase):
    def test_schema_numeric_fields_have_valid_ranges(self):
        schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))
        for key in ("p0", "p_max", "ramp_steps", "judge_timeout"):
            item = schema[key]
            self.assertIn(item["type"], ("int", "float"), key)
            sl = item.get("slider")
            self.assertIsNotNone(sl, f"{key} 缺少 slider 范围限制")
            self.assertLess(sl["min"], sl["max"])
            self.assertTrue(sl["min"] <= item["default"] <= sl["max"], key)
        for key in ("p0", "p_max"):
            self.assertEqual((schema[key]["slider"]["min"], schema[key]["slider"]["max"]), (0, 1))
        self.assertLessEqual(schema["ramp_steps"]["slider"]["max"], core.MAX_RAMP_STEPS)
        self.assertEqual(schema["judge_timeout"]["slider"]["min"], 1)
        self.assertLessEqual(schema["judge_timeout"]["slider"]["max"], 300)

    def test_schema_defaults_are_valid(self):
        schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))
        params, warns = core.sanitize_params(schema["p0"]["default"], schema["p_max"]["default"],
                                             schema["ramp_steps"]["default"])
        self.assertEqual(warns, [])
        self.assertEqual(params, core.DEFAULT_CUSTOM)
        self.assertIn(schema["intensity"]["default"], schema["intensity"]["options"])
        self.assertIn(schema["inject_mode"]["default"], schema["inject_mode"]["options"])


GARBAGE_CONFIG = {
    "enabled": "true", "intensity": "custom",
    "p0": "abc", "p_max": None, "ramp_steps": NAN,
    "judge_timeout": "x", "judge_provider": 123, "judge_prompt": 5,
    "judge_include_user": "false", "judge_only_when_injected": "yes",
    "keyword_hints": "喵\n汪", "scope_mode": "specified",
    "scope_sessions": [{"a": 1}, 5, None, "u1"], "inject_mode": "bogus",
}


class PluginWithBadConfig(unittest.IsolatedAsyncioTestCase):
    async def _commands(self, plugin, umo="u1"):
        out = {}
        for name in ("cmd_prob", "cmd_stats", "cmd_status", "cmd_reset", "cmd_clear"):
            out[name] = [x async for x in getattr(plugin, name)(make_event(umo))][-1]
            self.assertIsInstance(out[name], str, name)
        return out

    async def test_garbage_config_is_sanitized(self):
        p = make_plugin(FakeContext(delay=0.0), **GARBAGE_CONFIG)
        self.assertEqual(p._params(), core.DEFAULT_CUSTOM)
        self.assertEqual(p._timeout(), 30)
        self.assertEqual(p._scope_sessions(), {"5", "u1"})
        self.assertEqual(p._keywords(), ["喵", "汪"])
        self.assertEqual(p._inject_mode(), "temp_part")
        self.assertEqual(p._text("judge_provider"), "")
        self.assertTrue(p._flag("enabled", False))
        self.assertFalse(p._flag("judge_include_user", True))
        self.assertTrue(p._in_scope("u1"))
        self.assertFalse(p._in_scope("other"))

    async def test_garbage_config_never_breaks_hooks_or_commands(self):
        p = make_plugin(FakeContext(answer="是", delay=0.0), **GARBAGE_CONFIG)
        for _ in range(30):
            ev = make_event("u1")
            await p.on_req(ev, make_req())
            await p.on_resp(ev, make_resp())
            await asyncio.gather(*p._tasks)
        await self._commands(p)
        status =[x async for x in p.cmd_status(make_event("u1"))][0]
        self.assertIn("p0", status)
        self.assertIn("判定超时", status)
        self.assertIn("⚠", status)

    async def test_warnings_are_logged_once(self):
        p = make_plugin(FakeContext(), **GARBAGE_CONFIG)
        for _ in range(5):
            p._params()
        self.assertEqual(len(p._warned), 3)            # p0 / p_max / 上升步数 各一条，不重复刷屏

    async def test_bool_strings_and_enabled_off(self):
        p = make_plugin(FakeContext(), enabled="false")
        req = make_req()
        await p.on_req(make_event(), req)
        self.assertEqual(req.prompt, "你好")
        self.assertEqual(p._pending, {})

    async def test_non_string_instruction_means_disabled(self):
        for v in (123, None, ["x"], {"a": 1}, "   "):
            p = make_plugin(FakeContext(), style_instruction=v)
            req = make_req()
            await p.on_req(make_event(), req)
            self.assertEqual(req.prompt, "你好", repr(v))
            status = [x async for x in p.cmd_status(make_event())][0]
            self.assertIn("风格指令为空", status)

    async def test_specified_scope_with_empty_list_warns(self):
        p = make_plugin(FakeContext(), scope_mode="specified", scope_sessions=[])
        status = [x async for x in p.cmd_status(make_event())][0]
        self.assertIn("会话列表为空", status)

    async def test_hooks_swallow_internal_errors(self):
        p = make_plugin(FakeContext())
        bad_event = SimpleNamespace()                   # 没有 unified_msg_origin，访问会抛 AttributeError
        await p.on_req(bad_event, make_req())            # 不应向外抛
        await p.on_resp(bad_event, make_resp())

    async def test_corrupted_storage_does_not_crash(self):
        p = make_plugin(FakeContext(answer="否", delay=0.0))
        p._kv.update({"state:u1": ["bad"], "stats": "junk", "sessions": "junk"})
        ev = make_event("u1")
        await p.on_req(ev, make_req())
        await p.on_resp(ev, make_resp())
        await asyncio.gather(*p._tasks)
        self.assertEqual(p._states["u1"].k, 2)
        self.assertEqual(p._stats["messages"], 1)
        await self._commands(p)

        p2 = make_plugin(FakeContext(delay=0.0))
        p2._kv.update({"stats": {"messages": "x", "hits": -4, "injected": None,
                                 "intervals": [3, "a", None, -2, 0, 5.5, 4], "fp": 12}})
        s = await p2._get_stats()
        self.assertEqual((s["messages"], s["hits"], s["injected"]), (0, 0, 0))
        self.assertEqual(s["intervals"], [3, 4])
        self.assertIsInstance(s["fp"], str)

    async def test_selftest_count_argument(self):
        for arg, want in [("abc", 3), (None, 3), (2.5, 3), (-5, 1), (0, 1), (999, 10), (4, 4)]:
            p = make_plugin(FakeContext(answer="否", delay=0.0))
            out = [x async for x in p.cmd_selftest(make_event(), arg)]
            self.assertEqual(out[-1].count("】用户："), want, repr(arg))


class ChangeSettingsWithoutClearing(unittest.IsolatedAsyncioTestCase):
    async def _round(self, p, umo="u1"):
        ev = make_event(umo)
        await p.on_req(ev, make_req())
        await p.on_resp(ev, make_resp())
        await asyncio.gather(*p._tasks)

    async def test_switching_tiers_mid_run_is_safe(self):
        p = make_plugin(FakeContext(answer="否", delay=0.0), intensity="low")
        for _ in range(20):
            await self._round(p)
        k_before = p._states["u1"].k
        self.assertEqual(k_before, 21)
        for tier in ("high", "mid", "custom", "low", "bogus"):
            p.config["intensity"] = tier
            for _ in range(3):
                await self._round(p)
            for name in ("cmd_prob", "cmd_stats", "cmd_status"):
                text = [x async for x in getattr(p, name)(make_event())][-1]
                self.assertIsInstance(text, str)
        self.assertEqual(p._states["u1"].k, k_before + 15)   # k 照常累加，不受设置变化影响
        self.assertEqual(p._stats["messages"], 35)

    async def test_stale_stats_suppress_misleading_comparison(self):
        p = make_plugin(FakeContext(answer="否", delay=0.0), intensity="low")
        for _ in range(40):
            await self._round(p)
        before = [x async for x in p.cmd_stats(make_event())][0]
        self.assertIn("触发次数：预期", before)
        p.config["intensity"] = "high"                      # 改设置，不清空
        after = [x async for x in p.cmd_stats(make_event())][0]
        self.assertNotIn("触发次数：预期", after)             # 不给出预期值
        self.assertNotIn("检验：z", after)                   # 不做检验
        self.assertIn("不做预期对比和检验", after)
        self.assertIn("回复 40 条", after)                  # 原始计数仍然展示
        [x async for x in p.cmd_clear(make_event())]
        cleared = [x async for x in p.cmd_stats(make_event())][0]
        self.assertNotIn("不做预期对比和检验", cleared)      # 清空后恢复正常
        self.assertIn("回复 0 条", cleared)

    async def test_changing_instruction_marks_stale(self):
        p = make_plugin(FakeContext(answer="否", delay=0.0))
        await self._round(p)
        self.assertFalse(await p._stats_stale())
        p.config["style_instruction"] = "换一句指令"
        self.assertTrue(await p._stats_stale())

    async def test_large_k_after_lowering_ramp_is_fine(self):
        p = make_plugin(FakeContext(answer="否", delay=0.0), intensity="custom", p0=0.01, p_max=0.9, ramp_steps=500)
        p._states["u1"] = core.SessionState(k=400)
        p.config["ramp_steps"] = 5                           # 步数从 500 缩到 5，k 远超 n
        self.assertEqual(core.prob_at(400, *p._params()), 0.9)
        await self._round(p)
        self.assertEqual(p._states["u1"].k, 401)


if __name__ == "__main__":
    unittest.main()
