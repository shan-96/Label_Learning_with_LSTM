import re
import string

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from scrap.GlobalVars import COMMENT_SEP


def remove_punctuation(comment):
    comment = comment.translate(str.maketrans('', '', string.punctuation)).lower()
    return re.sub(r'\d+', '', process(comment))


def process(comment):
    lemma = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(comment)
    filtered_sentence = ""
    for w in tokens:
        if not w in stop_words:
            filtered_sentence = filtered_sentence + lemma.lemmatize(w) + " "
    return filtered_sentence


class Cleaner:
    def explode_row(self, row):
        return row[0], row[1], row[2]

    def getComments(self, comments):
        return comments[2:-2]

    def clean(self, comments):
        cleaned_comments = ""
        for comment in comments.split(COMMENT_SEP):
            comment = remove_punctuation(comment)
            comment = process(comment)
            cleaned_comments = cleaned_comments + comment

        return cleaned_comments
