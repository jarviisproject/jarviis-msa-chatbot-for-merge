import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing
from chat.preprocess import Preprocess


class IntentModel:
    def __init__(self, model_name, proprocess):
        self.MAX_SEQ_LEN = 15
        self.labels = {0 : 'weather', 1: 'suggestion', 2: 'todo'}

        self.model= load_model(model_name)

        self.p = proprocess

    def predict_class(self, question):
        # 형태소 분석
        pos = self.p.pos(question)

        # 문장 내 키워드 추출(불용어 제거)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]

        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=self.MAX_SEQ_LEN, padding='post')

        predict = self.model.predict(padded_seqs)
        predict_class = tf.math.argmax(predict, axis=1)
        return predict_class.numpy()[0]


if __name__ == '__main__':
    p = Preprocess(word2index_dic='./model/chatbot3_dict.bin', userdic='./model/user_nng.tsv')

    intent = IntentModel(model_name='./model/intent_model.h5', proprocess=p)

    question = '내일 뭐할까?'

    predict = intent.predict_class(question)
    predict_label = intent.labels[predict]

    print(question)
    print(f'의도 예측 클래스 : {predict}')
    print(f'의도 예측 레이블 : {predict_label}')


