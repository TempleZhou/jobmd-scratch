# jobmd.cn 爬虫

> 使用 python 3.x

## 环境初始化

```bash
pip3 install -r requirements.txt
```

## 数据爬取

```bash
cd Jobmd
python3 run.py
```

## 准备语料库

```bash
cd LevelPredictor
python3 prepare_corpora.py  # 纯语料库，生成词向量使用
python3 prepare_corpora_label.py  # 带标签的，供分类使用
```

## 训练出词向量

```bash
cd LevelPredictor
python3 train_nlp2.py
```

## 基于 SVM 预测出职位级别

```bash
cd LevelPredictor
python3 train_svm2.py
```