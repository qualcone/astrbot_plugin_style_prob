"""风险检查中发现的问题的回归测试。"""
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_flow import FakeContext, make_event, make_plugin, make_req, make_resp, core  # noqa: E402


class ParseYesNo(unittest.TestCase):
    def test_true_positives(self):
        for t in ["是", "是。", "  是的", "对", "有", "Yes", "yes.", "TRUE", "y", "「是」"]:
            self.assertIs(core.parse_yes_no(t), True, t)

    def test_true_negatives(self):
        for t in ["否", "不是", "不", "没有", "无", "No", "no.", "false", "n"]:
            self.assertIs(core.parse_yes_no(t), False, t)

    def test_not_misread(self):
        # 以前这些会被前缀匹配误判
        for t in ["是否出现了该风格？", "对不起，我无法判断", "yesterday it rained", "note: unclear",
                  "nothing", "", None, "也许"]:
            self.assertIsNone(core.parse_yes_no(t), t)


class BuildPrompt(unittest.TestCase):
    def test_placeholders_in_content_are_not_reexpanded(self):
        out = core.build_judge_prompt("指令:{instruction}\n回复:{reply}", "A", "回复里写了 {instruction} 和 {user}")
        self.assertIn("回复里写了 {instruction} 和 {user}", out)
        self.assertEqual(out.count("A"), 1)


class RoundLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_hit_reset_keeps_messages_that_arrived_during_judging(self):
        ctx = FakeContext(answer="是", delay=0.3)
        plugin = make_plugin(ctx, judge_only_when_injected=True)
        plugin._states["u1"] = core.SessionState(k=5)
        ev = make_event()
        await plugin.on_req(ev, make_req())          # 第 5 条，注入
        await plugin.on_resp(ev, make_resp())        # k -> 6，后台判定中
        for _ in range(2):                           # 判定期间又来了 2 条（不注入的会话外路径用 p0=0 模拟）
            plugin._states["u1"].k += 1
        await plugin._wait_tasks()
        # 触发那条是第 5 条：之后已过 2 条，新的 k 应为 3，而不是 1
        self.assertEqual(plugin._states["u1"].k, 3)

    async def test_hit_reset_never_below_one(self):
        ctx = FakeContext(answer="是", delay=0.1)
        plugin = make_plugin(ctx)
        plugin._states["u1"] = core.SessionState(k=5)
        ev = make_event()
        await plugin.on_req(ev, make_req())
        await plugin.on_resp(ev, make_resp())
        plugin._states["u1"].k = 1                   # 判定期间被 /风格重置
        await plugin._wait_tasks()
        self.assertEqual(plugin._states["u1"].k, 1)

    async def test_pending_holds_event_so_ids_cannot_collide(self):
        plugin = make_plugin(FakeContext())
        ev = make_event()
        await plugin.on_req(ev, make_req())
        info = plugin._pending[id(ev)]
        self.assertIs(info.ev, ev)

    async def test_judge_concurrency_is_bounded(self):
        active, peak = 0, 0

        class Ctx(FakeContext):
            async def llm_generate(self, **kw):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.05)
                active -= 1
                return type("R", (), {"completion_text": "是"})()

        plugin = make_plugin(Ctx())
        results = await asyncio.gather(*[plugin._judge("u1", "喵", "") for _ in range(12)])
        self.assertTrue(all(results))
        self.assertLessEqual(peak, 4)

    async def test_judge_failure_warning(self):
        plugin = make_plugin(FakeContext())
        stats = await plugin._get_stats()
        stats["judge_failed"], stats["judged"] = 6, 1
        self.assertTrue(any("判定失败" in w for w in plugin._config_warnings()))
        stats["judged"] = 10
        self.assertFalse(any("判定失败" in w for w in plugin._config_warnings()))


if __name__ == "__main__":
    unittest.main()
