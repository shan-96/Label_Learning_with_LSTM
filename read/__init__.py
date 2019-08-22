import csv

from read.Cleaner import Cleaner
from scrap import Writer
from scrap.GlobalVars import DATAFILENAME

cleaner = Cleaner()
writer = Writer(DATAFILENAME)

with open(DATAFILENAME, "rb") as csvfile:
    row = csv.reader(csvfile, delimiter=',')
    jira, comments, resolution = cleaner.explodeRow(row)
    comments = cleaner.getComments(comments)
    comments = cleaner.clean(comments)
    writer.write(jira, comments, resolution, csv.QUOTE_NONE)
