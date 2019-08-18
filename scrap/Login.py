import scrap.GlobalVars as GlobalVars


class Login:
    def login(self, session):
        # create auth payload
        payload = {"os_username": GlobalVars.USERNAME, "os_password": GlobalVars.PASSWORD}

        # perform login
        result = session.post(GlobalVars.LOGIN_URL, data=payload, headers=dict(referer=GlobalVars.LOGIN_URL))

        return result
