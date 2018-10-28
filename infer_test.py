#python example to infer document vectors from trained doc2vec model
import gensim.models as gm
import codecs
import numpy as np

#parameters
model = "toy_data/model/wiki_en_doc2vec.model.bin"
test_docs = "toy_data/test.txt" # test.txt为需要向量化的文本
output_file = "toy_data/test_vector.txt" #得到测试文本的每一行的向量表示

# 超参
start_alpha = 0.01
infer_epoch = 1000

#加载模型
m = gm.Doc2Vec.load(model)
test_docs = [x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines()]

#infer test vectors
output = open(output_file, "w")
for d in test_docs:
    output.write(" ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n")
output.flush()
output.close()
#print(len(test_docs)) #测试文本的行数

print(m.most_similar("party", topn=5)) # 找到与party单词最相近的前5个

#保存为numpy形式
test_vector = np.loadtxt('toy_data/test_vector.txt')
test_vector = np.save('toy_data/test_vector', test_vector)