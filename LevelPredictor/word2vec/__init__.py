# 参考 https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
from __future__ import absolute_import
from __future__ import print_function

import collections
import pickle

import jieba
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
# 判断是否全为中文
from sklearn.decomposition import PCA

from config import ROOT_DIR


def is_all_chinese(text):
    for _char in text:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def read_stop_words():
    # 读取停用词
    stop_words = []
    with open(f'{ROOT_DIR}/corpora_data/stop_words.txt', "r", encoding="UTF-8") as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))
    return stop_words


# 读取数据
def read_data():
    try:
        with open(f'{ROOT_DIR}/corpora_data/raw_words.bin', "rb") as f:
            lines = pickle.load(f)
            return lines
    except FileNotFoundError:
        pass

    stop_words = read_stop_words()

    # 读取文本，预处理，分词，得到词典
    raw_words_list = []
    with open(f'{ROOT_DIR}/corpora_data/corpora.txt', "r", encoding='UTF-8') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n', '')
            while ' ' in line:
                line = line.replace(' ', '')
            if len(line) > 0:  # 如果句子非空
                raw_words = list(jieba.cut(line, cut_all=False))
                sentence = []
                for raw_word in raw_words:
                    if raw_word not in stop_words and is_all_chinese(raw_word):
                        sentence.append(raw_word)
                raw_words_list.append(sentence)
            line = f.readline()

    with open(f'{ROOT_DIR}/corpora_data/raw_words.bin', "wb") as f:
        pickle.dump(raw_words_list, f)
    with open(f'{ROOT_DIR}/corpora_data/raw_words.txt', "w") as f:
        f.writelines(["\t".join(sentence) + "\n" for sentence in raw_words_list])
    return raw_words_list


# 每个词打上唯一编号
def build_dataset(vocabulary_size, words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


# 词向量可视化
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     fontproperties=FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'),
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)


# 降维可视化
def plot_pca_scatter(X_train, y_train):
    estimator = PCA(n_components=2)
    X_pca = estimator.fit_transform(X_train)
    colors = ['blue', 'red']
    for i in range(len(colors)):
        px = X_pca[np.array(y_train) == i, 0]
        py = X_pca[np.array(y_train) == i, 1]
        plt.scatter(px, py, c=colors[i])
    plt.legend(np.arange(0, 10).astype(str))
    plt.savefig('pca.png')


# 计算余弦相似度
def cosine_similarity(x, y, norm=True):
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos
