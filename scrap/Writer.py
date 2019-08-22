import csv


class Writer:
    filename = ""

    def __init__(self, filename):
        self.filename = filename

    def write(self, jira, comments, resolution, quote_rule):
        with open(self.filename, 'a', newline='') as csvfile:
            file_writer = csv.writer(csvfile, delimiter=',', escapechar=',', quoting=quote_rule)
            file_writer.writerow([jira, comments, resolution])
