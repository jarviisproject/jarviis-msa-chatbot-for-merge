import datetime
import requests
from icecream import ic


def get_suggestion():
    today = datetime.date.today()
    tom = today + datetime.timedelta(days=1)
    url = f'http://192.168.219.105:8000/api/suggestion/list'
    response = requests.get(url)
    data = response.json()
    return data


def get_todolist():
    today = datetime.date.today()
    tom = today + datetime.timedelta(days=1)
    url = f'http://192.168.219.105:8000/api/event/list'
    response = requests.get(url)
    data = response.json()
    return data


if __name__ == '__main__':
    get_todolist()