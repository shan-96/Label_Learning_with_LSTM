# Entry point for all my unit tests
import unittest

from read import Cleaner
from scrap.GlobalVars import COMMENT_SEP


class TestUtil(unittest.TestCase):
    def test_clean_comments(self):
        test_comments = "comment.1?" + COMMENT_SEP + "comment!2"
        clean_comments = "comment comment "
        c = Cleaner()
        actual = c.clean(test_comments)
        self.assertEqual(clean_comments, actual)


if __name__ == '__main__':
    unittest.main()
