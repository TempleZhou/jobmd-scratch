#! -*- coding: utf-8 -*-
import os
import pickle

import jieba
import numpy as np
from gensim.models import word2vec
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from LevelPredictor.config import ROOT_DIR
from LevelPredictor.word2vec import read_stop_words, is_all_chinese

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
    stop_words = read_stop_words()
    for (text, label) in tqdm(corpora_label):
        sentance = []
        raw_words = list(jieba.cut(text, cut_all=False))
        for raw_word in raw_words:
            if raw_word not in stop_words and is_all_chinese(raw_word):
                sentance.append(model.wv[raw_word])
        if len(sentance) > 0:
            avg_vec = np.sum(sentance, axis=0)/len(sentance)
            X.append(avg_vec.tolist())
            Y.append(label)

    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.3, random_state=11)

    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_train)
    print(accuracy_score(y_predict, y_train))

    y_predict = clf.predict(X_valid)
    print(accuracy_score(y_predict, y_valid))
