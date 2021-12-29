from django.db import models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow_datasets as tfds
import tensorflow as tf


# transformer 생성
class Transformer:
    def __init__(self):
        self.max_length = 20

    def excute(self, ):
        train_data = pd.read_csv('./data/intent_data.csv')
        print(train_data.head())
        answers = []
        questions = []
        for sentence in train_data['Q']:
            sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
            sentence = sentence.strip()
            questions.append(sentence)

        for sentence in train_data['label']:
            # 구두점에 대해서 띄어쓰기
            # ex) 12시 땡! -> 12시 땡 !
            sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
            sentence = sentence.strip()
            answers.append(sentence)


        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions, target_vocab_size=2**13)
        print(tokenizer)

        # 시작 토큰과 종료 토큰에 대한 정수 부여.
        START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

        # 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
        VOCAB_SIZE = tokenizer.vocab_size + 2

        print('시작 토큰 번호 :', START_TOKEN)
        print('종료 토큰 번호 :', END_TOKEN)
        print('단어 집합의 크기 :', VOCAB_SIZE)

        string = tokenizer.encode(questions[114])
        print(string)
        original_string = tokenizer.decode(string)
        print(original_string)

        tokenized_inputs, tokenized_outputs = [], []

        for (sentence1, sentence2) in zip(questions, answers):
            # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
            sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
            sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

        # 패딩
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=self.max_length, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=self.max_length, padding='post')

        print(tokenized_inputs)


if __name__ == '__main__':
    tr = Transformer()
    tr.excute()