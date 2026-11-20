"""内置判定模板：判定“这一条回复里有没有出现该风格”，忽略需要跨多条消息才能判断的规则。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_flow import core  # noqa: E402

STYLE = ("三种爱用的句式。一是「我说啊……」，引出一句正经话。二是用反问来吐槽，比如「那样不好吗？」。"
         "三是被追问时偶尔装傻，比如「是这样吗」「有吗」「算了」。这三种轮着用，同一种不要连着出现")


class BuiltInTemplate(unittest.TestCase):
    def build(self, **kw):
        return core.build_judge_prompt("", kw.get("instruction", STYLE), kw.get("reply", "我说啊，这事得认真。"),
                                       kw.get("user", ""))

    def test_instruction_and_reply_each_appear_once(self):
        p = self.build()
        self.assertEqual(p.count(STYLE), 1)
        self.assertEqual(p.count("我说啊，这事得认真。"), 1)
        self.assertNotIn("{instruction}", p)
        self.assertNotIn("{reply}", p)
        self.assertNotIn("{user}", p)

    def test_cross_message_rules_must_be_ignored(self):
        p = self.build()
        for phrase in ("轮流使用", "不要连续重复", "一律忽略", "不能因为它们在单条回复里无法验证"):
            self.assertIn(phrase, p)

    def test_judges_presence_not_full_compliance(self):
        p = self.build()
        self.assertIn("任意一种", p)
        self.assertIn("只有回复里完全看不出", p)
        self.assertIn("意思和用法一致即可", p)
        self.assertIn("不要评价回复本身的好坏", p)

    def test_explains_the_background(self):
        p = self.build()
        self.assertIn("背景", p)
        self.assertIn("临时追加", p)            # 风格说明是临时追加到这一轮的
        self.assertIn("统计", p)                # 判定结果用来统计出现频率
        self.assertIn("只依据回复内容本身", p)

    def test_template_is_generic(self):
        # 内置模板不能带有任何具体风格的内容，只靠 {instruction} 传入
        for specific in ("喵", "我说啊", "反问", "口头禅", "撒娇"):
            self.assertNotIn(specific, core.DEFAULT_JUDGE_PROMPT)

    def test_answer_format_kept(self):
        p = self.build()
        self.assertIn("只回答一个字：是 或 否", p)            # 测试替身靠这句话识别判定调用，也是解析所依赖的格式

    def test_user_message_block_is_optional(self):
        with_user = self.build(user="你刚才是不是在敷衍我？")
        self.assertIn("用户的话", with_user)
        self.assertIn("你刚才是不是在敷衍我？", with_user)
        self.assertNotIn("用户的话", self.build())

    def test_reply_with_placeholder_like_text_is_not_reexpanded(self):
        p = self.build(reply="我说啊 {instruction} {user}")
        self.assertIn("我说啊 {instruction} {user}", p)
        self.assertEqual(p.count(STYLE), 1)

    def test_custom_template_still_overrides(self):
        p = core.build_judge_prompt("只看有没有喵：{reply}", STYLE, "喵", "")
        self.assertEqual(p, "只看有没有喵：喵")

    def test_template_is_not_absurdly_long(self):
        self.assertLess(len(core.DEFAULT_JUDGE_PROMPT), 800)  # 判定每次都要付费，模板别太长


if __name__ == "__main__":
    unittest.main()
