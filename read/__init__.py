import csv

from read.Cleaner import Cleaner
from scrap import Writer
from scrap.GlobalVars import DATAFILENAME, FILENAME

if __name__ == "__main__":
    cleaner = Cleaner()
    writer = Writer(DATAFILENAME)

    with open(FILENAME, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            jira, comments, resolution = cleaner.explode_row(row)
            if (jira != "jira"):
                comments = cleaner.getComments(comments)
                comments = cleaner.clean(comments)
            writer.write(jira, comments, resolution, csv.QUOTE_NONE)
            if (jira != "jira"):
                print("Entry inserted for CXL-" + jira)
