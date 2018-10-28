#python example to train doc2vec model (with or without pre-trained word embeddings)

import gensim.models as gm
import logging

#doc2vec 参数
vector_size = 256 # 词向量长度，默认为100
window_size = 15 # 窗口大小，表示当前词与预测词在一个句子中的最大距离是多少
min_count = 1 #可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
sampling_threshold = 1e-5 # 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
negative_size = 5 #如果>0,则会采用negativesampling，用于设置多少个noise words（一般是5-20）
train_epoch = 100 # 迭代次数
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 1 # 用于控制训练的并行数

#pretrained word embeddings
pretrained_emb = "toy_data/pretrained_word_embeddings.txt" #None if use without pretrained embeddings

#输入语料库
train_corpus = "toy_data/wiki_en.txt"

#模型输出
save_model_name = 'wiki_en_doc2vec.model'
saved_path = "toy_data/model/wiki_en_doc2vec.model.bin"

#获取日志信息
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#训练 doc2vec 模型
docs = gm.doc2vec.TaggedLineDocument(train_corpus) #加载语料
model = gm.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count,\
                   hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=pretrained_emb, iter=train_epoch)

#保存模型
model.save(saved_path)
