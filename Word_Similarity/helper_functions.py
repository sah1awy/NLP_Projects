from __future__ import print_function,division
from builtins import range
import numpy as np 
from sklearn.metrics.pairwise import pairwise_distances
import re 
import nltk 
import spacy
from nltk.stem import WordNetLemmatizer,LancasterStemmer,PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import word_tokenize,sent_tokenize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tqdm import tqdm

print("Loading word vectors...")
word2vec = {}
embedding = []
idx2word = []
na = []
with open(r"D:\glove.840B.300d\glove.840B.300d.txt",'r') as f:
    for line in tqdm(f):
        try:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:],dtype=np.float32)
            word2vec[word] = vec
            embedding.append(vec)
            idx2word.append(word)
        except:
            na.append(line.split()[0])
            continue
print("Found %s word vectors"%len(idx2word))
embedding = np.array(embedding)    
V,D = embedding.shape


def scrap(link):
    page = requests.get(link)
    src = page.content
    soup = BeautifulSoup(src,'html.parser')
    match = soup.find('div',{"class":"section1"})
    match = match.find('div',{"class":"Normal"})
    for strong_tag in match.find_all('strong'):
        strong_tag.decompose()
    txt = match.get_text()
    return txt

def process(text):
    i = 0
    txt = ""
    sents = sent_tokenize(text)
    while i <= 5:
        txt += sents[i]
        i+=1
    txt = re.sub("[^A-Za-z]",' ',txt)
    txt = re.sub(r"\s\w\s","",txt)
    if txt[1] == " ":
        txt = txt[1:]
    txt = re.sub("\s+",' ',txt)
    txt = txt.lower()
    txt = txt.strip()
    nlp = spacy.load("en_core_web_lg")
    l = []
    sw = stopwords.words("english")
    doc = nlp(txt)
    for w in doc:
        if w.text not in sw:
            l.append(w.lemma_)
    res = list(pd.Series(l).value_counts().index[:5])
    return res

# Euclidean Distance
def dist1(a,b):
    return np.linalg.norm(a-b)

# Cosine Distance -> 1 - cosine Similarity
def dist2(a,b):
    return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

dist, metric = dist2, "euclidean"

def find_analogy(w1,w2,w3):
    for w in (w1,w2,w3):
        if w not in idx2word:
            print(f"{w} doesn't exist in the Dictionary.")
            return
    v1 = word2vec[w1]
    v2 = word2vec[w2]
    v3 = word2vec[w3]
    v0 = v1 - v2 + v3
    distances = pairwise_distances(v0.reshape(1,D),embedding,metric=metric).reshape(V)
    idxs = distances.argsort()[:4]
    for idx in idxs:
        if idx2word[idx] not in (w1,w2,w3):
            best_word = idx2word[idx]
            break
    print(w1, "-", w2, "=", best_word, "-", w3)

def nearest_neighbors(w,n=5):
    if w not in idx2word:
        print(f"{w} doesn't exist in the Dictionary.")
        return
    v = word2vec[w]
    distances = pairwise_distances(v.reshape(1,D),embedding,metric=metric).reshape(V)
    idxs = distances.argsort()[1:n+1]
    print("neighbors of: %s" % w)
    for idx in idxs:
        print("\t%s" % idx2word[idx])