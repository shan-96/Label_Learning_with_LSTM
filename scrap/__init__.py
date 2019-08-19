import requests

import scrap.GlobalVars as GlobalVars
from scrap.Document import Document
from scrap.Login import Login
from scrap.Writer import Writer

session = requests.session()
l = Login()
result = l.login(session)

if result.status_code == 200:
    document = Document()
    for ticket_num in range(GlobalVars.TICKET_START, GlobalVars.TICKET_END + 1):
        url = GlobalVars.URL + str(ticket_num)
        comments, resolution = document.scrap(session, url)
        if(resolution != ""):
            writer = Writer()
            writer.write(ticket_num, comments, resolution)
        print("Entry for CXL-" + str(ticket_num) + " inserted")
else:
    print("Could not log in")
    exit(-1)

session.close()

exit(0)
