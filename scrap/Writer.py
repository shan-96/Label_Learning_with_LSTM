import csv

from scrap.GlobalVars import FILENAME


class Writer:
    def write(self, jira, comments, resolution):
        with open(FILENAME, 'a', newline='') as csvfile:
            file_writer = csv.writer(csvfile, delimiter=',', escapechar=',', quoting=csv.QUOTE_ALL)
            file_writer.writerow([jira, comments, resolution])
