# 参考 https://www.jianshu.com/p/ce630c198762

import logging
import os

from gensim.models import word2vec

from word2vec import read_data, cosine_similarity

if __name__ == '__main__':
    sens_list = read_data()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if os.path.isfile("word2vec/word2vec.model"):
        model = word2vec.Word2Vec.load("word2vec/word2vec.model")
    else:
        # sg=0 使用 CBOW 模型 sg=1 使用 skip-gram
        model = word2vec.Word2Vec(sens_list, sg=0, size=128, min_count=1, iter=20, batch_words=10000)
        model.save("word2vec/word2vec.model")

    print(f'病人 vec: {model.wv["病人"]}')
    print(f'患者 vec: {model.wv["患者"]}')
    print(f'主任 vec: {model.wv["主任"]}')

    a = model.wv["病人"].tolist()
    b = model.wv["患者"].tolist()
    c = model.wv["主任"].tolist()

    print(f'病人:患者 余弦相似度 {cosine_similarity(a, b)}')
    print(f'病人:主任 余弦相似度 {cosine_similarity(a, c)}')
