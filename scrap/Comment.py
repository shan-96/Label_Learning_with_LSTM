from scrap.GlobalVars import COMMENT_END
from scrap.GlobalVars import COMMENT_SEP
from scrap.GlobalVars import COMMENT_START


class Comment:
    def getAllComments(self, soup):
        comments_string = COMMENT_START
        comments = soup.find_all("div", {"class": "activity-comment"})
        for comment in comments:
            comment_contents = comment.find_all("div", {"class": "action-body"})
            for comment_content in comment_contents:
                comments_string = comments_string + str(comment_content.text.encode("utf-8"))
                comments_string = comments_string + COMMENT_SEP
        comments_string = comments_string + COMMENT_END

        return comments_string
