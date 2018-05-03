from pymongo import MongoClient
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import nltk
from nltk import tokenize
from config import ROOTHPATH

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
client = MongoClient("localhost:4321")
pub = client.pub.publications
db=client.pub

es = Elasticsearch([{'host': 'localhost', 'port': 9200}],timeout=30, max_retries=10, retry_on_timeout=True)

es.cluster.health(wait_for_status='yellow', request_timeout=1)
papernames=[]
###############################
file=open(ROOTHPATH+'/data/allcorpus_papers.txt','r')
text = file.read()
sentences = tokenize.sent_tokenize(text)
count=0
docLabels = []
actions=[]
for i,sent in enumerate(sentences):
    try:
        neighbors=sentences[i+1]
        neighbor_count=count+1
    except:

            neighbors = sentences[i -1]
            neighbor_count = count - 1

    docLabels.append(count)

    actions.append({
                                   "_index": "devtwosentnew",
                                   "_type": "devtwosentnorulesnew",
                                   "_id":count,

                                   "_source" : {
                                       "content.chapter.sentpositive" : sent,
                                       "content.chapter.sentnegtive": neighbors,
                                       "neighborcount":neighbor_count




                                   }})
    count = count + 1

print(len(sentences))
print(len(docLabels))



res = helpers.bulk(es, actions)
print(res)