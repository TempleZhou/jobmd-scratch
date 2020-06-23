import pickle

from gensim.models import word2vec

from LevelPredictor.config import ROOT_DIR
from train_svm2 import tokenizer
from word2vec import read_stop_words

if __name__ == '__main__':
    nlp_model = word2vec.Word2Vec.load(f"{ROOT_DIR}/word2vec/word2vec.model")

    with open(f"{ROOT_DIR}/clf.model", "rb") as f:
        clf = pickle.load(f)

    my_sentences = ["1.爱岗敬业，身体健康，品行端正，事业心强，具有较强的沟通、组织、协调能力，有良好的职业道德和团队合作精神 2.全日制本科及以上学历，及以上职称",
                    "1、内科、肾内科、风湿免疫相关专业 2、硕士及以上学历 3、合同制聘用，北京户口有进编的机会4、医师及以上职称"]

    for my_sentence in my_sentences:
        vec_avg = [tokenizer(nlp_model, my_sentence)]
        level = clf.predict(vec_avg)
        print(level)
