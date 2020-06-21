#! -*- coding: utf-8 -*-

import json
import numpy as np
from sklearn.model_selection import train_test_split
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 基本信息
maxlen = 512
epochs = 20
batch_size = 8
learning_rate = 2e-5

model_name = 'chinese_L-12_H-768_A-12'
config_path = f'{model_name}/bert_config.json'
checkpoint_path = f'{model_name}/bert_model.ckpt'
dict_path = f'{model_name}/vocab.txt'

exp_labels = [1, 2, 3, 4]

label_map = {}
for (i, label) in enumerate(exp_labels):
    label_map[label] = i


def load_data(filename):
    D = []
    i = 0
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
            D.append((l['text'], l['label']))
            i += 1
    return D


# 读取数据
corpora_data = load_data('corpora_data/corpora_label')

train_data, valid_data = train_test_split(corpora_data, test_size=0.3, random_state=42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool=True
)

output = Dense(len(exp_labels), activation="sigmoid")(model.output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=Adam(learning_rate),
    metrics=['accuracy']
)


class DataGenerator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            label_ids = np.array([0.] * len(exp_labels))
            # label_ids into one-hot
            label_ids[label_map[label]] = 1.

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(label_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = np.array(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def extract_arguments(text):
    """arguments抽取函数
    """
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 510:
        tokens.pop(-2)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    labels = model.predict([[token_ids], [segment_ids]])[0]

    return np.array(exp_labels)[np.where(labels > 0.5)[0].tolist()]


def evaluate(data):
    """评测函数（跟官方评测结果不一定相同，但很接近）
    """
    # TP: 预测为1(Positive)，实际也为1(Truth - 预测对了)
    # TN: 预测为0(Negative)，实际也为0(Truth - 预测对了)
    # FP: 预测为1(Positive)，实际为0(False - 预测错了)
    # FN: 预测为0(Negative)，实际为1(False - 预测错了)
    TP, TN, FP, FN = 0, 0, 0, 0
    for text, inv_arguments in tqdm(data):
        pred_arguments = extract_arguments(text)
        # 预测值
        if inv_arguments in pred_arguments:
            TP += 1
        else:
            FP += 1
            FN += 1

    precision, recall = TP / (TP + FP), TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall


def predict_to_file(in_file, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            arguments = extract_arguments(l['text'])
            l['pre_label'] = arguments
            l = json.dumps(l, ensure_ascii=False)
            fw.write(l + '\n')
    fw.close()


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """

    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('best_model_event.weights')
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


if __name__ == '__main__':
    train_generator = DataGenerator(train_data[:100], batch_size)
    evaluator = Evaluator()

    batch_token_ids = []
    batch_segment_ids = []
    batch_labels = []
    for job_desc in train_data[:100]:
        print(job_desc)
        text = job_desc[0]
        label = job_desc[1]
        token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
        label_ids = np.array([0.] * len(exp_labels))
        # label_ids into one-hot
        label_ids[label_map[label]] = 1.

        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append(label_ids)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )