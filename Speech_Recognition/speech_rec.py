from nltk import word_tokenize as w
import nltk
def find_noun(t):
    l = nltk.pos_tag(w(t))
    nouns = []
    print(l)
    for (a,b) in l :
        if b == "NN" or b == "NNS":
            nouns.append(a)
            print(a)
    return nouns





