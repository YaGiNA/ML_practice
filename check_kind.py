
# coding: utf-8

# In[1]:


import os
import numpy
import pandas as pd


# In[2]:


train_file = os.path.abspath('./liar_dataset/train.tsv')
test_file = os.path.abspath('./liar_dataset/test.tsv')
valid_file = os.path.abspath('./liar_dataset/valid.tsv')


# In[3]:


def read_all_tsv(filenames):
    all_df = pd.concat([
        pd.read_table(
            filename, delimiter='\t', header=None, names=(
            'api_id', 't-f', 'quote', 'subject', 'speaker', 'job', 'city', 'affriation',
            'h_pof', 'h_false', 'h_mostly_false', 'h_half_true', 'h_mostly_true', 'platform'
        ))
        for filename in filenames
    ])
    return all_df


# In[ ]:


def all_tf_nums(filenames):
    df = read_all_tsv(filenames)
    all_false = df[df["t-f"].fillna("").str.contains("barely|false|pants")].shape[0]
    all_true = df[df["t-f"].fillna("").str.contains("half|mostly")].shape[0] + df[df["t-f"].fillna("")=="true"].shape[0]
    return all_true, all_false


# In[4]:


df = read_all_tsv([train_file, test_file, valid_file])


# In[43]:


df[df["platform"].fillna("").str.contains("tweet|website|online")]


# In[44]:


df.shape
# df[df.isnull().any(axis=1)]
# df[df["platform"].isnull()]


# In[22]:


df["platform"].unique()

