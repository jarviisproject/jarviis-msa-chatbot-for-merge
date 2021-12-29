# 챗봇 사전 파일 생성
from tensorflow.keras import preprocessing
import pickle

from chat.preprocess import Preprocess


class Create_corpus:
    def __init__(self):
        pass

    def read_corpus_data(self, filename):
        with open(filename, 'r', encoding='UTF-8') as f:
            data = [line.split('\t') for line in f.read().splitlines()]
            data = data[1:]
        return data

    def corpus(self):
        # 말뭉치 데이터 가져오기
        corpus_data = self.read_corpus_data('./data/intent_data.csv')

        #말뭉치 데이터에서 키워드만 추출해서 사전 리스트 생성
        p = Preprocess()
        dict = []
        for c in corpus_data:
            print(c)
            pos = p.pos(c[0])
            for k in pos:
                dict.append(k[0])

        tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')
        tokenizer.fit_on_texts(dict)
        word_index = tokenizer.word_index

        f = open('./data/chatbot3_dict.bin', 'wb')
        try:
            pickle.dump(word_index, f)
        except Exception as e:
            print(e)
        finally:
            f.close()


def test_corpus():
    f = open('./data/chatbot3_dict.bin', 'rb')
    word_index = pickle.load(f)
    f.close()

    sent = '개발자 자격증 추천해줘'

    p = Preprocess(userdic='../data/user_nng.tsv')

    pos = p.pos(sent)

    keywords = p.get_keywords(pos, without_tag=True)
    for word in keywords:
        try:
            print(word, word_index[word])
        except KeyError:
            print(word, word_index['OOV'])


if __name__ == '__main__':
    cor = Create_corpus()
    # cor.corpus()
    test_corpus()
