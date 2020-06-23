#! -*- coding: utf-8 -*-
import os
import pickle

import jieba
import numpy as np
from gensim.models import word2vec
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from LevelPredictor.config import ROOT_DIR
from LevelPredictor.word2vec import read_stop_words, is_all_chinese

# 加载停用词
stop_words = read_stop_words()


def tokenizer(nlp_model, txt):
    words = []
    raw_words = list(jieba.cut(txt, cut_all=False))
    for raw_word in raw_words:
        if raw_word not in stop_words and is_all_chinese(raw_word):
            words.append(nlp_model.wv[raw_word])

    if len(words) > 0:
        return (np.sum(words, axis=0) / len(words)).tolist()
    else:
        return None


if __name__ == '__main__':
    with open('corpora_data/corpora_label.bin', "rb") as f:
        corpora_label = pickle.load(f)

    model = None
    # 加载模型
    if os.path.isfile(f"{ROOT_DIR}/word2vec/word2vec.model"):
        model = word2vec.Word2Vec.load(f"{ROOT_DIR}/word2vec/word2vec.model")
    else:
        raise Exception

    X = []
    Y = []
    for (text, label) in tqdm(corpora_label):
        sentence = tokenizer(model, text)
        if sentence:
            X.append(sentence)
            Y.append(label)

    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.3)

    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    with open(f"{ROOT_DIR}/clf.model", "wb") as f:
        pickle.dump(clf, f)

    y_predict = clf.predict(X_train)
    print(classification_report(y_train, y_predict, target_names=['新手', '老手'], digits=4))

    y_predict = clf.predict(X_valid)
    print(classification_report(y_valid, y_predict, target_names=['新手', '老手'], digits=4))
