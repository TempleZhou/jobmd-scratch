# 参考 https://www.jianshu.com/p/ce630c198762

from gensim.models import word2vec
import pandas as pd
import logging
import os

from gensim.models.callbacks import CallbackAny2Vec

from LevelPredictor.word2vec import read_data, cosine_similarity


class Callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_previous = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_previous
        self.loss_to_be_previous = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1


if __name__ == '__main__':
    sens_list = read_data()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if os.path.isfile("word2vec.model"):
        model = word2vec.Word2Vec.load("word2vec.model")
    else:
        # sg=0 使用 CBOW 模型 sg=1 使用 skip-gram
        model = word2vec.Word2Vec(sens_list, sg=1, size=128, min_count=1, iter=20, batch_words=10000,
                                  compute_loss=True, callbacks=[Callback()])
        model.save("word2vec.model")

    print(f'病人 vec: {model.wv["病人"]}')
    print(f'患者 vec: {model.wv["患者"]}')
    print(f'主任 vec: {model.wv["主任"]}')

    a = model.wv["病人"].tolist()
    b = model.wv["患者"].tolist()
    c = model.wv["主任"].tolist()

    print(f'病人:患者 余弦相似度 {cosine_similarity(a, b)}')
    print(f'病人:主任 余弦相似度 {cosine_similarity(a, c)}')
