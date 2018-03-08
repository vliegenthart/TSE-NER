import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence
from os import listdir
from nltk import tokenize, sent_tokenize
from os.path import isfile, join
from config import ROOTHPATH
from gensim.models import Doc2Vec
from elasticsearch import Elasticsearch


file=open('/Users/sepidehmesbah/PycharmProjects/NERDetector/data/allcorpus_papers.txt','r')
text = file.read()
sentences = tokenize.sent_tokenize(text)
count=0
docLabels = []
for i in range(0,len(sentences)):
    docLabels.append(count)
    count=count+1
print(len(sentences))
print(len(docLabels))




class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc.split(),tags=[self.labels_list[idx]])
it = LabeledLineSentence(sentences, docLabels)



model = gensim.models.Doc2Vec(size=100, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(it)
for epoch in range(10):
    model.train(it, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(it,total_examples=model.corpus_count, epochs=model.iter)

model.save(ROOTHPATH+'/data/doc2vec.model')
