from __future__ import absolute_import
import os
import numpy as np
import re

def get_story(file,only_supporting=False):
    data = []
    story = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.lower()
            index,line = line.split(' ',1)#将序号和文本分开
            index = int(index)
            if index == 1:
                story = []
            if '\t' in line:#说明是问句和答案行
                q,ans,support = line.split('\t')
                q = tokenize(q)#分词
                ans = [ans]
                if only_supporting:#data只有和support相关的story
                    support = map(int,support.split())
                    substory = [story[i-1] for i in support]#substory
                else:
                    substory = [x for x in story if x]
                data.append((substory,q,ans))#data统一在这里append
                story.append('')
            else:
                sent = tokenize(line)
                story.append(sent)
    return data

def tokenize(line):
    return line[:-2].strip().split(' ')#将.和？移除

def load_task(dir_name,task_id,only_supporting=False):
    files = os.listdir(dir_name)
    files = [os.path.join(dir_name,f) for f in files]#列出所有文件名
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]#筛选符合条件的文件
    test_file = [f for f in files if s in f and 'test' in f][0]#筛选符合条件的文件
    train_data = get_story(train_file,only_supporting)
    test_data = get_story(test_file,only_supporting)
    return train_data,test_data

def vectorize_data(data,word_idx,sentence_size,memory_size):
    '''
    将story和query和answer向量化
    '''
    S = []
    Q = []
    A = []
    for story,query,answer in data:
        ss = []
        for i,sentence in enumerate(story,1):
            ls = max(0,sentence_size-len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)#sentence句子长度不够的话 补零
        ss = ss[::-1][:memory_size][::-1]

        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - memory_size - i + len(ss)#将句子向量的最后一位变成time word
        #填充memory
        lm = max(0,memory_size-len(ss))
        for _ in range(lm):
            ss.append([0]*sentence_size)

        lq = max(0,sentence_size-len(query))#填充query
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx)+1)#把answer变成一个one-hot向量
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)

    return np.array(S),np.array(Q),np.array(A)
















