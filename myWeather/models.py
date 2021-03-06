import datetime as dt
from datetime import datetime

import requests

# Create your tests here.


class Weather(object):
    def __init__(self):
        pass

    def process(self):
        return self.weather_now()

    def weather_now(self):
        base_date, base_time, nx, ny = self.weather_api()
        url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
        params = {
            'serviceKey': 'Azs8t5lc5MVlv67BnCCXpwjGc9eLdowYV2q3MZO0wqwOkWo3vpUP0PsHimKL0osXusxlIb+888C2E+PquYdixQ==',
            'pageNo': '1',
            'numOfRows': '10',
            'dataType': 'JSON',
            'base_date': base_date,
            'base_time': base_time,
            'nx': f'{nx}',
            'ny': f'{ny}'}
        response = requests.get(url, params=params)
        weather = response.json().get('response').get('body').get('items')
        weather = list(weather.values())[0]
        for i in weather:
            if i['category'] == 'SKY':
                sky = i
            elif i['category'] == 'PTY':
                pty = i
        # print(f'SKY dict : {sky}')
        # print(f'PTY dict : {pty}')
        # print(f'SKY result : {sky["fcstValue"]}')
        # print(f'PTY result : {pty["fcstValue"]}')
        return self.weather_transfer(sky,pty)

    def weather_pre(self):
        base_date, base_time, nx, ny = self.weather_api()
        url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
        params = {
            'serviceKey': 'Azs8t5lc5MVlv67BnCCXpwjGc9eLdowYV2q3MZO0wqwOkWo3vpUP0PsHimKL0osXusxlIb+888C2E+PquYdixQ==',
            'pageNo': '1',
            'numOfRows': '1000',
            'dataType': 'JSON',
            'base_date': base_date,
            'base_time': base_time,
            'nx': f'{nx}',
            'ny': f'{ny}'}
        response = requests.get(url, params=params)
        # print(response.url)
        weather = response.json().get('response').get('body').get('items')
        weather = list(weather.values())[0]
        time = str((int(base_time) + 100)) if int(base_time) + 100 < 2400 else "0000"
        time = time if len(time) == 4 else f'0{time}'
        day = [base_date, self.date_string(datetime.now() + dt.timedelta(days=1)), self.date_string(datetime.now() + dt.timedelta(days=2))]
        sky, pty = [], []
        for i in weather:
            if i['fcstDate'] == base_date and i['fcstTime'] == time:
                if i['category'] == 'SKY':
                    sky.append(i)
                elif i['category'] == 'PTY':
                    pty.append(i)
            elif i['fcstDate'] == day[1] and i['fcstTime'] == time:
                if i['category'] == 'SKY':
                    sky.append(i)
                elif i['category'] == 'PTY':
                    pty.append(i)
            elif i['fcstDate'] == day[2] and i['fcstTime'] == time:
                if i['category'] == 'SKY':
                    sky.append(i)
                elif i['category'] == 'PTY':
                    pty.append(i)
        result = [self.weather_transfer(sky[i], pty[i]) for i in range(len(sky))]
        return {j : result[i] for i, j in enumerate(day)}

    def weather_api(self):
        # ????????? ?????????
        # nx = 61
        # ny = 126
        # ?????? ??????
        nx = 60
        ny = 127
        # ?????? ??????
        now = datetime.now()
        base_date = self.date_string(now)
        # 1??? ??? 8??? ???????????? ???????????? ??????.(0200, 0500, 0800, 1100, 1400, 1700, 2000, 2300)
        # ?????? api??? ??????????????? ????????? ?????? ????????? ??????????????? ???????????? base_time, base_date??? ??????
        if now.hour < 2 or (now.hour == 2 and now.minute <= 10):  # 0???~2??? 10??? ??????
            base_time = "2300"
        elif now.hour < 5 or (now.hour == 5 and now.minute <= 10):  # 2??? 11???~5??? 10??? ??????
            base_time = "0200"
        elif now.hour < 8 or (now.hour == 8 and now.minute <= 10):  # 5??? 11???~8??? 10??? ??????
            base_time = "0500"
        elif now.hour <= 11 or now.minute <= 10:  # 8??? 11???~11??? 10??? ??????
            base_time = "0800"
        elif now.hour < 14 or (now.hour == 14 and now.minute <= 10):  # 11??? 11???~14??? 10??? ??????
            base_time = "1100"
        elif now.hour < 17 or (now.hour == 17 and now.minute <= 10):  # 14??? 11???~17??? 10??? ??????
            base_time = "1400"
        elif now.hour < 20 or (now.hour == 20 and now.minute <= 10):  # 17??? 11???~20??? 10??? ??????
            base_time = "1700"
        elif now.hour < 23 or (now.hour == 23 and now.minute <= 10):  # 20??? 11???~23??? 10??? ??????
            base_time = "2000"
        else:  # 23??? 11???~23??? 59???
            base_time = "2300"
        return base_date, base_time, nx, ny

    def date_string(self, date):
        return f'{date.year}{date.month if date.month > 9 else f"0{date.month}"}{date.day if date.day > 9 else f"0{date.day}"}'

    def weather_transfer(self, sky, pty):
        return self.sky_transfer(sky["fcstValue"]) if self.pty_transfer(pty["fcstValue"]) == '0' else self.pty_transfer(
            pty["fcstValue"])

    # ????????????(SKY) ?????? : ??????(1), ????????????(3), ??????(4)
    def sky_transfer(self, sky):
        if sky == '1':
            return '??????'
        elif sky == '3':
            return '?????? ??????'
        elif sky == '4':
            return '??????'
        else:
            return '????????? ??????'

    # ????????????(PTY) ?????? : (??????) ??????(0), ???(1), ???/???(2), ???(3), ?????????(4)
    def pty_transfer(self, pty):
        if pty == '0':
            return '0'
        elif pty == '1':
            return '???'
        elif pty == '2':
            return '???, ???'
        elif pty == '3':
            return '???'
        elif pty == '4':
            return '?????????'
        else:
            return '????????? ??????'





