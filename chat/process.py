import datetime

from myWeather.models import Weather
import datetime as dt
import requests
import json
from django.http import JsonResponse
from chat.intentModel import IntentModel
from chat.preprocess import Preprocess


def weather_question(question):
    w = Weather()
    q = question['chatAnswer']
    if "오늘" in q:
        today = dt.datetime.now() + dt.timedelta()
        tom = str(today)[0:4] + str(today)[5:7] + str(today)[8:10]
        return w.weather_pre()[tom]
    elif "내일" in q:
        today = dt.datetime.now() + dt.timedelta(days=1)
        tom = str(today)[0:4] + str(today)[5:7] + str(today)[8:10]
        return w.weather_pre()[tom]
    elif "모레" in q:
        today = dt.datetime.now() + dt.timedelta(days=2)
        tom = str(today)[0:4] + str(today)[5:7] + str(today)[8:10]
        return w.weather_pre()[tom]
    else:
        return "날씨를 알 수 없습니다."


def todo_answer(question):
    q = question['chatAnswer']
    f = open('chat/data/todo_data.json', encoding='UTF-8')
    # twoday = datetime.date.today() + datetime.timedelta(days=2)
    # tom = datetime.date.today() + datetime.timedelta(days=1)
    # today = datetime.date.today()
    twoday = '2021-12-22'
    tom = "2021-12-21"
    today = "2021-12-20"
    data = json.load(f)
    titles = []
    newlist = []
    if '오늘' in q:
        for i in data:
            if str(i['start']).find(today) > -1:
                titles.append(i['title'])
    elif '내일' in q:
        for i in data:
            if str(i['start']).find(tom) > -1:
                titles.append(i['title'])
    elif '모레' in q:
        for i in data:
            if str(i['start']).find(twoday) > -1:
                titles.append(i['title'])
    if len(titles) == 0:
        return "일정이 없습니다."
    else:
        for v in titles:
            if v not in newlist:
                newlist.append(v)
        return f'todo 일정입니다 {newlist}'




def suggestions_answer(question):
    q = question['chatAnswer']
    f = open('chat/data/suggestions.json', encoding='UTF-8')
    data = json.load(f)
    titles = []
    newlist = []
    if '개발자' in q:
        for i in data:
            if i['classification'] == 'DEV':
                titles.append(i['title'])
    elif '다이어트' in q:
        for i in data:
            if i['classification'] == 'DEV':
                titles.append(i['title'])
    else:
        for i in data:
            if i['classification'] == 'DEV':
                titles.append(i['title'])
    if len(titles) == 0 :
        return "추천 목록이 없습니다."
    else:
        for v in titles:
            if v not in newlist:
                newlist.append(v)
        return f'추천 목록 입니다 {newlist}'


# def todo_answer(question):
#     date = None
#     url = f'http://192.168.219.105:8000/api/event/{date}'
#     toodlist = []
#     q = question['question']
#     if "오늘" in q:
#         date = dt.date.today()
#         tom = date + dt.timedelta(days=0)
#         response = requests.get(url)
#         data = response.json()
#         for i in data:
#             toodlist.append(i['title'])
#         return toodlist
#     elif '내일' in q:
#         today = dt.date.today()
#         tom = today + dt.timedelta(days=1)
#         url = f'http://192.168.219.105:8000/api/suggestion/list'
#         response = requests.get(url)
#         data = response.json()
#         for i in data:
#             toodlist.append(i['title'])
#         return toodlist
#     elif '모레' in q:
#         today = dt.date.today()
#         tom = today + dt.timedelta(days=2)
#         url = f'http://192.168.219.105:8000/api/suggestion/list'
#         response = requests.get(url)
#         data = response.json()
#         for i in data:
#             toodlist.append(i['title'])
#         return toodlist
#     else :
#         url = f'http://192.168.219.105:8000/api/suggestion/list'
#         return ''


class IntentChat:
    def __init__(self):
        self.p = Preprocess(word2index_dic='chat/model/chatbot3_dict.bin', userdic='chat/model/user_nng.tsv')
        self.intent = IntentModel(model_name='chat/model/intent_model.h5', proprocess=self.p)

    def predictModel(self, question):
        return self.intent.predict_class(question)

    def predic_label(self, predict):
        return self.intent.labels[self.predictModel(predict)]


if __name__ == '__main__':
    print(datetime.date.today())