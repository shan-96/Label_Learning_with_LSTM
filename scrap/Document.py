from bs4 import BeautifulSoup

from scrap.Comment import Comment


class Document:
    def scrap(self, session, url, ticket_num):
        # print(ticket_num)
        content = session.get(url, headers=dict(referer=url)).text
        soup = BeautifulSoup(content, 'lxml')
        comments = Comment()
        comments_string = comments.getAllComments(soup)
        resolution = soup.find("span", {"id": "resolution-val"})
        resolution_string = " ".join(resolution.text.split())
        # print(resolution_string)
        # print(comments_string)
        return comments_string, resolution_string
