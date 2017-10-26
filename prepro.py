'''

'''

import numpy as np
import pandas as pd
import re
import bs4 as bs
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class preprocess:

    def __init__(self, file_, cols, stop_flag=True):
        self.file_ = file_
        self.stop_flag = stop_flag
        self.cols = cols
        self.cntr = 0
        if stop_flag:
            self.stop_words = set(stopwords.words('english'))

    def pri(self):
        print(self.file_)
        print(self.cols)
        print(self.stop_flag)

    def stop_remover(self,s):
        new_s = ''
        words = word_tokenize(s)
        for w in words:
            if w not in self.stop_words:
                new_s = new_s + ' ' + w
        return new_s[1:]

    # -*- coding: utf-8 -*-
    def isEnglish(self,s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    def collapse_terms(self,s):
        for w in word_tokenize(s):
            if (w.find('http')!=-1) or (w.find('watch')!=-1 and len(w)>9):
                s = s.replace(w,"_link_feature")
            if self.isEnglish(w) == False:
                s = s.replace(w,"")
        return s

    def read_and_clean(self):
        '''
            read data into pandas dataframe
            keep only columns that are required
        '''
        frames = [pd.read_csv(x) for x in self.file_]
        self.data = pd.concat(frames,axis=0,ignore_index=True)
        self.data = self.data[self.cols]
        print(self.data.shape)

        '''
            parse text using html decoder because some of the text are in html format.
        '''
        warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
        self.data['CONTENT'] = self.data['CONTENT'].apply(lambda x: bs.BeautifulSoup(x,'html.parser').get_text())

        '''
            remove punctuation
            remove stop words if stop_words flag is true
        '''
        self.data['CONTENT'] = self.data['CONTENT'].apply(lambda x : re.sub('[^A-Za-z ]','',x))
        if self.stop_flag:
            self.data['CONTENT'] = self.data['CONTENT'].apply(lambda x : self.stop_remover(x))

        '''
            collapse some of the feature containing links into one commom term
            also remove non english words.
        '''
        self.data['CONTETNT'] = self.data['CONTENT'].apply(lambda x :self. collapse_terms(x))


if __name__ == "__main__":
    # Testing area
    obj1 = preprocess(['f1.csv','f2.csv','f3.csv','f4.csv','f5.csv'],['CONTENT','CLASS'])
    obj1.pri()
    obj1.read_and_clean()
    print(obj1.data.head())
