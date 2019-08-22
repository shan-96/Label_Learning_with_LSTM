import string

import stopwords as stopwords

from scrap.GlobalVars import COMMENT_SEP


def filter_text(comment):
    ##TODO: complete this method
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    comment = re.sub('[^A-Za-z0-9]+', ' ', comment)
    stop_free = " ".join([i for i in comment.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized.lower()


class Cleaner:
    def explodeRow(self, row):
        one, two, three = row.split(',')
        return one, two, three

    def getComments(self, comments):
        return comments[2:-2]

    def clean(self, comments):
        cleaned_comments = ""
        for comment in comments.split(COMMENT_SEP):
            comment = filter_text(comment)
            cleaned_comments = comment + COMMENT_SEP

        return cleaned_comments
