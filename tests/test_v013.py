"""v0.1.3：会话 + 人设两个独立筛选、用户消息过滤、合并写入与清理。"""
import asyncio
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_flow import FakeContext, make_event, make_plugin, make_req, make_resp, core  # noqa: E402
from astrbot_plugin_style_prob import main as plugin_main  # noqa: E402


def ev(umo="u1", persona=None, msg="你好", extras=None):
    e = make_event(umo, msg)
    data = dict(extras or {})
    data["_trace_persona_id"] = persona
    e.get_extra = lambda k, d=None: data.get(k, d)
    return e


async def injected(plugin, event):
    req = make_req()
    await plugin.on_req(event, req)
    return "请撒娇" in req.prompt


class ScopeFilters(unittest.IsolatedAsyncioTestCase):
    async def test_empty_means_everything(self):
        plugin = make_plugin(FakeContext())
        self.assertTrue(await injected(plugin, ev("any", "anyone")))

    async def test_sessions_only(self):
        plugin = make_plugin(FakeContext(), scope_sessions=["u1", "u2"])
        self.assertTrue(await injected(plugin, ev("u1")))
        self.assertTrue(await injected(plugin, ev("u2")))
        self.assertFalse(await injected(plugin, ev("u3")))

    async def test_personas_multi(self):
        plugin = make_plugin(FakeContext(), scope_personas=["cat", "dog"])
        self.assertTrue(await injected(plugin, ev("a", "cat")))
        self.assertTrue(await injected(plugin, ev("b", "dog")))
        self.assertFalse(await injected(plugin, ev("c", "fish")))
        self.assertFalse(await injected(plugin, ev("d", None)))   # 取不到人设：不生效

    async def test_both_must_match(self):
        plugin = make_plugin(FakeContext(), scope_sessions=["u1"], scope_personas=["cat"])
        self.assertTrue(await injected(plugin, ev("u1", "cat")))
        self.assertFalse(await injected(plugin, ev("u1", "dog")))
        self.assertFalse(await injected(plugin, ev("u2", "cat")))

    async def test_wildcard_and_separators(self):
        plugin = make_plugin(FakeContext(), scope_personas="cat，dog\nfish")
        self.assertEqual(plugin._scope_personas(), {"cat", "dog", "fish"})
        self.assertTrue(make_plugin(FakeContext(), scope_personas=["*"])._in_scope("u", "x"))

    async def test_unknown_persona_id_warns(self):
        ctx = FakeContext()
        ctx.persona_manager = SimpleNamespace(
            personas=[SimpleNamespace(persona_id="cat"), SimpleNamespace(persona_id="default")],
            personas_v3=[])
        plugin = make_plugin(ctx, scope_personas=["cat", "cta"])
        warns = plugin._config_warnings()
        self.assertTrue(any("cta" in w and "不存在" in w for w in warns))
        self.assertFalse(any("cat、" in w or "：cat" in w for w in warns))


class LegacyMigration(unittest.IsolatedAsyncioTestCase):
    def test_all_clears_leftover_lists(self):
        p = make_plugin(FakeContext(), scope_mode="all", scope_sessions=["u9"], scope_personas=[])
        self.assertEqual(p.config["scope_mode"], "")
        self.assertEqual(p._scope_sessions(), set())

    def test_specified_and_session_keep_sessions(self):
        for mode in ("specified", "session"):
            p = make_plugin(FakeContext(), scope_mode=mode, scope_sessions=["u1"], scope_persona="cat")
            self.assertEqual(p._scope_sessions(), {"u1"}, mode)
            self.assertEqual(p._scope_personas(), set(), mode)

    def test_persona_mode_moves_single_persona_into_list(self):
        p = make_plugin(FakeContext(), scope_mode="persona", scope_persona="cat", scope_sessions=["u1"])
        self.assertEqual(p._scope_personas(), {"cat"})
        self.assertEqual(p._scope_sessions(), set())
        self.assertEqual(p.config["scope_persona"], "")

    def test_no_legacy_key_means_no_migration(self):
        p = make_plugin(FakeContext(), scope_sessions=["u1"], scope_personas=["cat"])
        self.assertEqual((p._scope_sessions(), p._scope_personas()), ({"u1"}, {"cat"}))

    def test_save_config_called_when_available(self):
        class Cfg(dict):
            saved = False

            def save_config(self):
                Cfg.saved = True

        from astrbot_plugin_style_prob.main import StyleProbPlugin
        cfg = Cfg({"scope_mode": "all"})
        StyleProbPlugin(FakeContext(), cfg)
        self.assertTrue(Cfg.saved)


class UserTurnFilter(unittest.IsolatedAsyncioTestCase):
    async def test_normal_user_message_is_processed(self):
        plugin = make_plugin(FakeContext())
        self.assertTrue(await injected(plugin, ev(msg="你好")))

    async def test_background_and_subagent_events_skipped(self):
        plugin = make_plugin(FakeContext())
        self.assertFalse(await injected(plugin, ev("a", extras={"_background_stop_signal": object()})))
        self.assertFalse(await injected(plugin, ev("b", extras={"_subagent_wakeup": {"x": 1}})))
        self.assertEqual(plugin._pending, {})

    async def test_prompt_not_containing_user_message_skipped(self):
        plugin = make_plugin(FakeContext())
        req = make_req()
        req.prompt = "其他插件自己拼的提示词"
        await plugin.on_req(ev(msg="你好"), req)
        self.assertEqual(req.prompt, "其他插件自己拼的提示词")
        self.assertEqual(plugin._pending, {})

    async def test_empty_message_still_processed(self):
        plugin = make_plugin(FakeContext())        # 纯图片等没有文字的消息
        req = make_req()
        await plugin.on_req(ev(msg=""), req)
        self.assertIn("请撒娇", req.prompt)

    async def test_filter_can_be_turned_off(self):
        plugin = make_plugin(FakeContext(), only_user_requests=False)
        req = make_req()
        req.prompt = "别处拼的提示词"
        await plugin.on_req(ev(msg="你好"), req)
        self.assertIn("请撒娇", req.prompt)


class Persistence(unittest.IsolatedAsyncioTestCase):
    async def test_writes_are_coalesced(self):
        plugin = make_plugin(FakeContext(answer="否", delay=0.0))
        plugin.FLUSH_DELAY = 0.15
        writes = []
        orig = plugin.put_kv_data

        async def counting(k, v):
            writes.append(k)
            await orig(k, v)

        plugin.put_kv_data = counting
        for _ in range(10):
            e = ev("u1")
            await plugin.on_req(e, make_req())
            await plugin.on_resp(e, make_resp())
        await plugin._wait_tasks()
        self.assertEqual(writes.count("state:u1"), 1)     # 10 条回复只写一次
        self.assertEqual(writes.count("stats"), 1)
        self.assertEqual(plugin._kv["state:u1"]["k"], 11)  # 而且是最终值
        self.assertEqual(plugin._kv["stats"]["messages"], 10)

    async def test_terminate_flushes_pending_changes(self):
        plugin = make_plugin(FakeContext(answer="否", delay=0.0))
        plugin.FLUSH_DELAY = 30                            # 写入还没触发就退出
        e = ev("u1")
        await plugin.on_req(e, make_req())
        await plugin.on_resp(e, make_resp())
        self.assertNotIn("state:u1", plugin._kv)
        await plugin.terminate()
        self.assertEqual(plugin._kv["state:u1"]["k"], 2)

    async def test_failed_write_keeps_changes_for_retry(self):
        plugin = make_plugin(FakeContext(answer="否", delay=0.0))
        orig = plugin.put_kv_data
        fail = {"on": True}

        async def flaky(k, v):
            if fail["on"]:
                raise RuntimeError("disk full")
            await orig(k, v)

        plugin.put_kv_data = flaky
        e = ev("u1")
        await plugin.on_req(e, make_req())
        await plugin.on_resp(e, make_resp())
        await plugin._wait_tasks()
        self.assertIn("u1", plugin._dirty)                 # 失败：改动还在
        fail["on"] = False
        await plugin._flush()
        self.assertEqual(plugin._kv["state:u1"]["k"], 2)
        self.assertEqual(plugin._dirty, set())

    async def test_reset_command_writes_immediately(self):
        plugin = make_plugin(FakeContext())
        plugin.FLUSH_DELAY = 30
        plugin._states["u1"] = core.SessionState(k=9)
        _ = [x async for x in plugin.cmd_reset(make_event("u1"))]
        self.assertEqual(plugin._kv["state:u1"]["k"], 1)
        await plugin.terminate()

    async def test_memory_eviction_never_drops_unsaved_state(self):
        old_cap = plugin_main.MAX_STATES_IN_MEMORY
        plugin_main.MAX_STATES_IN_MEMORY = 10
        try:
            plugin = make_plugin(FakeContext())
            plugin.FLUSH_DELAY = 30
            now = time.time()
            for i in range(10):
                plugin._states[f"s{i}"] = core.SessionState(k=3, ts=now - 1000 + i)
            plugin._dirty.add("s0")                          # 最老的一个，但有未写入的改动
            await plugin._get_state("new")                   # 超过上限，触发淘汰
            self.assertLessEqual(len(plugin._states), 9)
            self.assertIn("s0", plugin._states)              # 没写入的不能丢
            self.assertIn("new", plugin._states)
            self.assertNotIn("s1", plugin._states)           # 其余最老的被淘汰
        finally:
            plugin_main.MAX_STATES_IN_MEMORY = old_cap

    async def test_evicted_state_reloads_from_storage(self):
        plugin = make_plugin(FakeContext())
        plugin._kv["state:u1"] = {"k": 7, "ts": time.time()}
        self.assertEqual((await plugin._get_state("u1")).k, 7)
        del plugin._states["u1"]
        self.assertEqual((await plugin._get_state("u1")).k, 7)

    async def test_prune_removes_only_long_idle_sessions(self):
        plugin = make_plugin(FakeContext())
        now = time.time()
        day = 86400
        plugin._kv.update({
            "sessions": ["old", "fresh", "legacy", "loaded"],
            "state:old": {"k": 3, "ts": now - 200 * day},
            "state:fresh": {"k": 4, "ts": now - 5 * day},
            "state:legacy": {"k": 5},                        # 旧版本写入的，没有 ts
            "state:loaded": {"k": 6, "ts": now - 300 * day},
        })
        plugin._states["loaded"] = core.SessionState(k=6)    # 内存里有：说明刚活跃过，不能清
        await plugin._maybe_prune()
        self.assertNotIn("state:old", plugin._kv)
        self.assertIn("state:fresh", plugin._kv)
        self.assertIn("state:loaded", plugin._kv)
        self.assertIn("ts", plugin._kv["state:legacy"])      # 补上了活跃时间，之后才能参与清理
        self.assertEqual(plugin._kv["sessions"], ["fresh", "legacy", "loaded"])

    async def test_prune_runs_at_most_once_per_interval(self):
        plugin = make_plugin(FakeContext())
        plugin._kv.update({"sessions": ["old"], "state:old": {"k": 1, "ts": 1.0}})
        await plugin._maybe_prune()
        plugin._kv.update({"sessions": ["old2"], "state:old2": {"k": 1, "ts": 1.0}})
        plugin._index = None
        await plugin._maybe_prune()                          # 间隔内再次调用：什么也不做
        self.assertIn("state:old2", plugin._kv)

    def test_session_state_ts_roundtrip_and_garbage(self):
        st = core.SessionState.from_dict({"k": "5", "ts": 1700000000})
        self.assertEqual((st.k, st.ts), (5, 1700000000.0))
        now = time.time()
        for bad in (None, "x", [], {"k": 2, "ts": "abc"}, {"k": 2, "ts": -5}, {"k": 2, "ts": now + 10 ** 9},
                    {"k": 2, "ts": float("nan")}):
            s = core.SessionState.from_dict(bad)
            self.assertGreaterEqual(s.k, 1)
            self.assertLessEqual(s.ts, time.time() + 1)
            self.assertGreater(s.ts, now - 5)


if __name__ == "__main__":
    unittest.main()
