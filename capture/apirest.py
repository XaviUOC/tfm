import requests
import urllib
import os


class ApiRequestException(Exception):
    def __init__(self, response):
        super(ApiRequestException, self).__init__(response)


class ApiRest:
    def __init__(self, base_url, **headers):
        self._base_url = base_url
        self._session = self._init_session(**headers)

    @property
    def base_url(self):
        return self._base_url

    @staticmethod
    def _init_session(**kwargs):
        session = requests.session()
        headers = {'Accept': 'application/json',
                   "Content-type": "application/x-www-form-urlencoded",
                   'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36',
                   'Accept-Language': 'en-US'}
        headers.update(kwargs)
        session.headers.update(headers)
        return session

    def get(self, url, **params):
        data = urllib.parse.urlencode(params)
        abs_url = os.path.join(self._base_url, url)
        response = requests.get(abs_url, data)
        if response.status_code == 200:
            return response.json()
        raise ApiRequestException(response)
