"""v0.1.7：日志里记录注入后模型的完整回复与判定模型的原始回答（可开关）。"""
import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_flow import FakeContext, make_event, make_plugin, make_req, make_resp, core  # noqa: E402
from test_v013 import ev  # noqa: E402
from astrbot_plugin_style_prob import main as plugin_main  # noqa: E402


class Capture:
    def __enter__(self):
        self.messages = []
        outer = self

        class H(logging.Handler):
            def emit(self, rec):
                outer.messages.append(rec.getMessage())

        self.h = H()
        self.lg = plugin_main.logger
        self.old = self.lg.level
        self.lg.addHandler(self.h)
        self.lg.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *a):
        self.lg.removeHandler(self.h)
        self.lg.setLevel(self.old)

    def lines(self):
        return [m for m in self.messages if "已注入" in m]


async def one_round(plugin, reply="喵～收到啦", umo="u1"):
    e = ev(umo)
    await plugin.on_req(e, make_req())
    await plugin.on_resp(e, make_resp(reply))
    await plugin._wait_tasks()


class FullReplyLog(unittest.IsolatedAsyncioTestCase):
    async def test_default_logs_only_snippet_and_raw_answer(self):
        with Capture() as cap:
            plugin = make_plugin(FakeContext(answer="否", delay=0.0))
            await one_round(plugin, reply="很长的回复。" * 20)
        (line,) = cap.lines()
        self.assertIn("判定模型回答='否'", line)
        self.assertIn("回复开头=", line)
        self.assertNotIn("完整回复=", line)

    async def test_switch_on_logs_full_reply(self):
        long_reply = "第一行\n第二行 " + "字" * 100
        with Capture() as cap:
            plugin = make_plugin(FakeContext(answer="否，没有出现该风格", delay=0.0), log_full_reply=True)
            await one_round(plugin, reply=long_reply)
        (line,) = cap.lines()
        self.assertIn("完整回复=", line)
        self.assertIn("第二行", line)
        self.assertIn("字" * 100, line)                       # 没被截成 30 个字
        self.assertNotIn("\n", line)                           # 换行被转义，日志仍是一行
        self.assertIn("判定模型回答='否，没有出现该风格'", line)

    async def test_full_reply_is_truncated_in_log(self):
        with Capture() as cap:
            plugin = make_plugin(FakeContext(answer="是", delay=0.0), log_full_reply=True)
            await one_round(plugin, reply="字" * (plugin_main.FULL_REPLY_LOG_CHARS + 500))
        (line,) = cap.lines()
        self.assertIn("已截断", line)
        self.assertLess(line.count("字"), plugin_main.FULL_REPLY_LOG_CHARS + 50)

    async def test_full_reply_switch_works_even_if_round_log_is_off(self):
        with Capture() as cap:
            plugin = make_plugin(FakeContext(answer="是", delay=0.0), log_rounds=False, log_full_reply=True)
            await one_round(plugin)
        self.assertEqual(len(cap.lines()), 1)
        with Capture() as cap:
            plugin = make_plugin(FakeContext(answer="是", delay=0.0), log_rounds=False, log_full_reply=False)
            await one_round(plugin)
        self.assertEqual(cap.lines(), [])

    async def test_non_injected_rounds_never_logged(self):
        with Capture() as cap:
            plugin = make_plugin(FakeContext(delay=0.0), p0=0.0, p_max=0.0, log_full_reply=True)
            await one_round(plugin)
        self.assertEqual(cap.lines(), [])


class RawAnswerInRecord(unittest.IsolatedAsyncioTestCase):
    async def test_raw_answer_stored_and_truncated(self):
        plugin = make_plugin(FakeContext(answer="否。" + "解释" * 100, delay=0.0))
        await one_round(plugin)
        raw = plugin._recent[0]["raw"]
        self.assertTrue(raw.startswith("否。"))
        self.assertLessEqual(len(raw), plugin_main.RAW_CHARS + 1)
        self.assertTrue(raw.endswith("…"))

    async def test_keyword_hit_is_labelled(self):
        plugin = make_plugin(FakeContext(answer="否", delay=0.0), keyword_hints=["喵"])
        await one_round(plugin, reply="喵～")
        self.assertEqual(plugin._recent[0]["result"], "hit")
        self.assertIn("关键词", plugin._recent[0]["raw"])

    async def test_call_failure_and_unparseable_answers_are_visible(self):
        class Boom(FakeContext):
            async def llm_generate(self, **kw):
                raise RuntimeError("secret internal detail")
        plugin = make_plugin(Boom())
        await one_round(plugin)
        rec = plugin._recent[0]
        self.assertEqual(rec["result"], "failed")
        self.assertIn("调用失败", rec["raw"])
        self.assertNotIn("secret", rec["raw"])                 # 不把内部异常细节带到页面

        weird = make_plugin(FakeContext(answer="也许吧", delay=0.0))
        await one_round(weird)
        self.assertEqual(weird._recent[0]["result"], "failed")  # 无法解析
        self.assertEqual(weird._recent[0]["raw"], "也许吧")

    async def test_snapshot_exposes_raw(self):
        plugin = make_plugin(FakeContext(answer="是", delay=0.0))
        await one_round(plugin)
        d = await plugin._snapshot()
        self.assertEqual(d["recent"][0]["raw"], "是")

    async def test_selftest_judge_call_still_works_without_out(self):
        plugin = make_plugin(FakeContext(answer="是", delay=0.0))
        self.assertIs(await plugin._judge("u1", "喵", ""), True)


if __name__ == "__main__":
    unittest.main()
