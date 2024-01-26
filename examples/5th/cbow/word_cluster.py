#!/usr/bin/env python3
# coding: utf-8
# File: word_cluster.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-4-25
import numpy as np

class WordCluster:
    def __init__(self):
        self.embedding_path = 'model/word2word_wordvec.bin'
        self.embedding_path = 'model/word2doc_wordvec.bin'
        self.embedding_path = 'model/skipgram_wordvec.bin'
        self.embedding_path = 'model/cbow_wordvec.bin'
        self.word_embedding_dict, self.word_dict, self.word_embeddings = self.load_model(self.embedding_path)
        self.similar_num = 10

    #加载词向量文件
    def load_model(self, embedding_path):
        print('loading models....')
        word_embedding_dict = {}
        word_embeddings = []
        word_dict = {}
        index = 0
        for line in open(embedding_path):
            line = line.strip().split('\t')
            word = line[0]
            word_embedding = np.array([float(item) for item in line[1].split(',') if item])
            word_embedding_dict[word] = word_embedding
            word_embeddings.append(word_embedding)
            word_dict[index] = word
            index += 1
        return word_embedding_dict, word_dict, np.array(word_embeddings)
    # 计算相似度
    def similarity_cosine(self, word):
        A = self.word_embedding_dict[word]
        B = (self.word_embeddings).T
        dot_num = np.dot(A, B)
        denom = np.linalg.norm(A) * np.linalg.norm(B)
        cos = dot_num / denom
        sims = 0.5 + 0.5 * cos
        sim_dict = {self.word_dict[index]: sim for index, sim in enumerate(sims.tolist()) if word != self.word_dict[index]}
        sim_words = sorted(sim_dict.items(), key=lambda asd: asd[1], reverse=True)[:self.similar_num]
        return sim_words
    #获取相似词语
    def get_similar_words(self, word):
        if word in self.word_embedding_dict:
            return self.similarity_cosine(word)
        else:
            return []

def test():
    vec = WordCluster()
    while 1:
        word = input('enter an word to search:').strip()
        simi_words = vec.get_similar_words(word)
        for word in simi_words:
            print(word)

test()