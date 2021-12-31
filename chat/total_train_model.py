import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate
from chat.preprocess import Preprocess


class Intent_model_create():
    def __init__(self):
        self.MAX_SEQ_LEN = 15

    def execute(self):
        train_file = './data/intent_data.csv'
        data = pd.read_csv(train_file, delimiter=',')
        question = data['Q'].tolist()
        intent = data['label'].tolist()

        p = Preprocess(word2index_dic='./model/chatbot3_dict.bin',
                       userdic='./model/user_nng.tsv')

        # 단어 시퀸스 생성
        sequences = []
        for sentence in question:
            pos = p.pos(sentence)
            keywords = p.get_keywords(pos, without_tag=True)
            seq = p.get_wordidx_sequence(keywords)
            sequences.append(seq)

        # 단어 인덱스 시퀸스 백터 생성
        # 단어 시퀸스 백터 크기
        print(sequences)
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=self.MAX_SEQ_LEN, padding='post')

        # 학습용, 검증용, 테스트용 데이터셋 생성
        ds = tf.data.Dataset.from_tensor_slices((padded_seqs,intent))
        ds = ds.shuffle(len(question))

        train_size = int(len(padded_seqs) * 0.7)
        val_size = int(len(padded_seqs) * 0.2)
        test_size = int(len(padded_seqs) * 0.1)

        train_ds = ds.take(train_size).batch(20)
        val_ds = ds.skip(train_size).take(val_size).batch(20)
        test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

        # 하이퍼파라미터 설정
        dropout_prob = 0.5
        EMB_SIZE = 128
        EPOCH = 5
        VOCAB_SIZE = len(p.word_index) + 1

        # CNN 모델 정의
        input_layer = Input(shape=(self.MAX_SEQ_LEN,))
        embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=self.MAX_SEQ_LEN)(input_layer)
        dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

        conv1 = Conv1D(filters=128, kernel_size=3, padding='valid', activation=tf.nn.relu)(dropout_emb)
        pool1 = GlobalMaxPool1D()(conv1)
        conv2 = Conv1D(filters=128, kernel_size=4, padding='valid', activation=tf.nn.relu)(dropout_emb)
        pool2 = GlobalMaxPool1D()(conv2)
        conv3 = Conv1D(filters=128, kernel_size=5, padding='valid', activation=tf.nn.relu)(dropout_emb)
        pool3 = GlobalMaxPool1D()(conv3)

        concat = concatenate([pool1, pool2, pool3])
        hidden = Dense(128, activation=tf.nn.relu)(concat)
        dropout_hidden = Dropout(rate=dropout_prob)(hidden)
        logits = Dense(3, name='logits')(dropout_hidden)
        predictions = Dense(3, activation=tf.nn.softmax)(logits)

        #  모델 생성
        model = Model(inputs=input_layer, outputs=predictions)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # 모델 학습
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

        # 모델 평가(테스트 데이터셋 이용)
        loss, accuracy = model.evaluate(test_ds, verbose=1)
        print(f'Accuracy :  {accuracy * 100}')
        print(f'loss : {loss}')

        # 모델 저장
        model.save('intent_model.h5')


if __name__ == '__main__':
    im = Intent_model_create()
    im.execute()


