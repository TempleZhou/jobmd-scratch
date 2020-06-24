# 参考 https://www.jianshu.com/p/ce630c198762

import logging
import os

from gensim.models import word2vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from LevelPredictor.word2vec import read_data, cosine_similarity

if __name__ == '__main__':
    sens_list = read_data()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if os.path.isfile("doc2vec.model"):
        model = word2vec.Word2Vec.load("word2vec.model")
    else:
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sens_list)]
        # sg=0 使用 CBOW 模型 sg=1 使用 skip-gram
        model = Doc2Vec(documents, vector_size=256, window=10, min_count=5,
                        workers=4, alpha=0.025, min_alpha=0.025, epochs=12)
        model.save("doc2vec.model")
