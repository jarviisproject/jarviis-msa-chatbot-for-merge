import pandas as pd
from pandas import DataFrame, Series


class Dataset:
    def __init__(self):
        pass

    # 데이터 질문과  label를 리스트로 전환
    def process_intent_file(self) -> list:
        file = pd.read_csv('../data/intent_data.csv', encoding='utf-8')
        questions = file['Q'].values.tolist()
        intents = file['label']
        print(questions)
        print(intents)
        print([(question, intents) for question in questions])
        return [(question, intents) for question in questions]

    # intent를 숫자로 변환
    def make_intent_dict(self) -> dict:
        file = pd.read_csv('../data/intent_data.csv', encoding='utf-8')
        intents = file['label']

        label_dict, index = {}, -1
        for intent in intents:
            if intent not in label_dict:
                index += 1
            label_dict[intent] = index
            print(label_dict)
        return label_dict

    def make_intent(self):
        pass



if __name__ == '__main__':
    d = Dataset()
    # d.process_intent_file()
    d.make_intent_dict()