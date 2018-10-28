# doc2vec
使用gensim库训练doc2vec模型

（1）下载wiki英文数据压缩包：https://dumps.wikimedia.org/enwiki/latest/

（2）在cmd下切换到这个压缩包文件存放的目录下，运行命令：
python process_wiki.py enwiki-latest-pages-articles1.xml-p10p30302.bz2 wiki.en.text得到wiki_en.txt

（3）训练 train_model.py

（4）测试 infer_test.py
