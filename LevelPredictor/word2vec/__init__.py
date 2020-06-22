# 参考 https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
from __future__ import absolute_import
from __future__ import print_function

import collections

import jieba
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# 判断是否全为中文
def is_all_chinese(text):
    for _char in text:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


# 读取数据
def read_data():
    try:
        with open('../corpora_data/raw_words.txt', "r", encoding="UTF-8") as f:
            lines = [word[:-1] for word in f.readlines()]
            return lines
    except FileNotFoundError:
        pass

    # 读取停用词
    stop_words = []
    with open('../corpora_data/stop_words.txt', "r", encoding="UTF-8") as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))

    # 读取文本，预处理，分词，得到词典
    raw_words_list = []
    with open('../corpora_data/corpora.txt', "r", encoding='UTF-8') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n', '')
            while ' ' in line:
                line = line.replace(' ', '')
            if len(line) > 0:  # 如果句子非空
                raw_words = list(jieba.cut(line, cut_all=False))
                for raw_word in raw_words:
                    if raw_word not in stop_words and is_all_chinese(raw_word):
                        raw_words_list.append(raw_word)
            line = f.readline()

    with open('../corpora_data/raw_words.txt', "w", encoding="UTF-8") as f:
        f.write("\n".join(raw_words_list))
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