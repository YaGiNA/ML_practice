
# coding: utf-8

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[3]:


train_file = os.path.abspath('./liar_dataset/train.tsv')
test_file = os.path.abspath('./liar_dataset/test.tsv')


# In[4]:


def get_tf_title(train_file):
    tfs, titles = [], []
    with open(train_file, "r") as f:
        line = f.readline()
        while line:
            cols = line.split("\t")
            raw_tf = cols[1]
            if raw_tf in {"false", "pants-fire"}:
                tf = False
            else:
                tf = True
            title = cols[2]
            tfs.append(tf)
            titles.append(title)
            line = f.readline()
    infos = [tfs, titles]     
    return infos


# In[46]:


train_infos, test_infos = get_tf_title(train_file), get_tf_title(test_file)
infos = [train_infos[0] + test_infos[0], train_infos[1] + test_infos[1]]


# http://moritamori.hatenablog.com/entry/tfidf_vectorizer
def make_tfidf_vec(infos):
    vec = TfidfVectorizer(max_df=10, ngram_range=(1, 1), sublinear_tf=True, norm='l2', stop_words='english')
    docs = infos[1]
    term_doc = vec.fit_transform(docs)
    info_x = term_doc.toarray()
    info_y = infos[0]
    return info_x, info_y


info_x, info_y = make_tfidf_vec(infos)
train_x, train_y = info_x[:len(train_infos[0])], info_y[:len(train_infos[1])]
test_x, test_y = info_x[len(train_infos[0]):], info_y[len(train_infos[1]):]
