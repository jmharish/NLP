
from nltk.tokenize import word_tokenize , sent_tokenize
import gensim
import numpy as np
import sys
f = open ("doc.txt","r")
t = f.read()
print(t)
l = sent_tokenize(t)
llw =[]
for i in l :
    llw.append([w.lower() for w in word_tokenize(i)])
f.close()
dic = gensim.corpora.Dictionary(llw)
print(dic.token2id)
corpus = [dic.doc2bow(i)for i in llw ]#gives the id of the words in each sentence along with the frequency of the word in that sentence 
print(corpus)
#To find the TFIDF frequency
tf_idf = gensim.models.TfidfModel(corpus)
for i in tf_idf[corpus]:
    print([(dic[id], np.around(freq, decimals=2)) for id, freq in i])# we use numpy only to round the decimal value can just do: print(i)
# gives the TFIDF value based on the number of time the word has been repeated in the document and the length of the line in which the word is present
sims = gensim.similarities.Similarity('C:\\Harish\\NLP' ,tf_idf[corpus],len(dic))
# creates an index for the document in the NLP/ directory
fq = open ("query_doc.txt","r+")
t = fq.read()
print(t)
l = sent_tokenize(t)
llw2=[]
fq.close()
for i in l :
    llw2.append([w.lower() for w in word_tokenize(i)])
corpus_query=[dic.doc2bow(c) for c in llw2]
query_tf_idf = tf_idf[corpus_query]# checks for  the words in previous document in each line of query document and gives the frequency of the words
print(sims[query_tf_idf])#checks the similarity of each line of query doc to each line of index doc and gives similatiry values individualy
avg =0 
for i in sims[query_tf_idf]:
    avg += sum (i)/len(i)
avg/=.03
print(avg,"% similar document")
