import prepro
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.metrics import  accuracy_score,f1_score,confusion_matrix

class NBC(object):

    def __init__(self,data,split_size=0.2,smoothing=True,smoothing_value=1):
        self.data = data
        self.smoothing = smoothing
        self.smoothing_value = smoothing_value
        self.split_size = split_size
        self.c1 = ''
        self.c2 = ''

    def feature_extraction(self):
        self.train_data,self.test_data = train_test_split(self.data,test_size=self.split_size,random_state=42)
        self.count_c1 = 0
        self.count_c2 = 0
        for idx,d in self.train_data.iterrows():
            if d['CLASS']==0:
                self.count_c1 +=1
                self.c1 = self.c1 + " " +d['CONTENT']
            else:
                self.count_c2 +=1
                self.c2 = self.c2 + " " +d['CONTENT']
        self.tokens_c1 = word_tokenize(self.c1)
        self.tokens_c2 = word_tokenize(self.c2)
        cnt_ = Counter(self.tokens_c1 + self.tokens_c2)
        self.vocab = [x for x,y in cnt_.items() if y>0]
        self.V = len(self.vocab)
        self.tokens_c1 = [w for w in self.tokens_c1 if w in self.vocab]
        self.tokens_c2 = [w for w in self.tokens_c2 if w in self.vocab]
        self.wordcount_c1 = len(self.tokens_c1)
        self.wordcount_c2 = len(self.tokens_c2)

    def train(self):
        self.freq_c1 = Counter(self.tokens_c1)
        self.freq_c2 = Counter(self.tokens_c2)
        self.prob_c1 = {}
        self.prob_c2 = {}
        for v in self.vocab:
            if v not in self.freq_c1:
                self.freq_c1[v] = 0
            if v not in self.freq_c2:
                self.freq_c2[v] = 0
            if self.smoothing:
                self.prob_c1[v] = (self.freq_c1[v]+self.smoothing_value)/float(self.wordcount_c1 + (self.V*self.smoothing_value))
                self.prob_c2[v] = (self.freq_c2[v]+self.smoothing_value)/float(self.wordcount_c2 + (self.V*self.smoothing_value))
            else:
                self.prob_c1[v] = (self.freq_c1[v])/float(self.wordcount_c1)
                self.prob_c2[v] = (self.freq_c2[v])/float(self.wordcount_c2)

    def predict(self,s):
        obb = prepro.preprocess()
        s = re.sub('[^A-Za-z ]','',s)
        s = obb.collapse_terms(s)
        input_ = word_tokenize(s)
        input_ = [w.lower() for w in input_ if w.lower() in self.vocab]
        pre_c1 = 1.0
        pre_c2 = 1.0
        for i in input_:
            pre_c1 *= self.prob_c1[i]
            pre_c2 *= self.prob_c2[i]
        if pre_c1*(self.count_c1/float(self.count_c1+self.count_c2)) < pre_c2*(self.count_c2/float(self.count_c1+self.count_c2)):
            return 1
        else:
            return 0

    #testing method
    def test_run(self):
        op = []
        for idx,d in self.test_data.iterrows():
            op.append(self.predict(d['CONTENT']))
        mat = confusion_matrix(self.test_data['CLASS'],op)
        acc = accuracy_score(self.test_data['CLASS'],op)
        fs = f1_score(self.test_data['CLASS'],op)
        print(mat)
        print("Accuracy : "+str(acc*100)+"%")
        print(fs)

if __name__ == "__main__":
    ob1 = prepro.preprocess(['f1.csv','f2.csv','f3.csv','f4.csv','f5.csv'],['CONTENT','CLASS'])
    data = ob1.read_and_clean()
    ob2 = NBC(data)
    ob2.feature_extraction()
    ob2.train()
    ob2.test_run()
