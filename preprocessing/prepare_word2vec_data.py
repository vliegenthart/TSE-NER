from pymongo import MongoClient
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import math
import nltk
import string
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
###############################

client = MongoClient('localhost:4321')
db=client.pub
pub = client.pub.publications
es = Elasticsearch(
    [{'host': 'localhost', 'port': 9200}], timeout=30, max_retries=10, retry_on_timeout=True
)
es.cluster.health(wait_for_status='yellow', request_timeout=1)
list_of_pubs=[]
def returnnames(mongo_string_search, db):
    # mongo_string_search = {"dblpkey": "{}".format(dblkey)}
    results = db.publications.find(mongo_string_search)
    chapters = list()
    chapter_nums = list()
    list_of_docs = list()
    # list_of_abstracts = list()
    merged_chapters = list()
    my_dict = {
        "dblpkey": "",

    }
    for i, r in enumerate(results):
        # try:
        # list_of_sections = list()
        my_dict['dblpkey'] = r['dblpkey']
        list_of_docs.append((my_dict))

        my_dict = {
            "dblpkey": "",

        }

    return list_of_docs
filter_conference = ["WWW", "ICSE", "VLDB", "PVLDB", "JCDL", "TREC",  "SIGIR", "ICWSM", "ECDL", "ESWC"]

for booktitle in filter_conference:
    mongo_string_search = {'$and': [{'booktitle': booktitle}, {'content.fulltext': {'$exists': True}}]}
    list_of_pubs.append(returnnames(mongo_string_search, db))
papersText = []
for pubs in list_of_pubs:

    for cur in pubs:

        query = {"query":
            {"match": {
                "_id": {
                    "query": cur['dblpkey'],
                    "operator": "and"
                }
            }
            }
        }

        res = es.search(index="ir", doc_type="allpubs",
                        body=query, size=200)

        for doc in res['hits']['hits']:
            # sentence = doc["_source"]["text"].replace(',', ' ')
            fulltext=doc["_source"]["text"]
            print(doc["_id"])
            print(fulltext)

            fulltext= fulltext.translate(str.maketrans('', '', string.punctuation))
            papersText.append(fulltext.lower())
papersText=" ".join(papersText)

f1 = open("files/word2vecData.txt", "w")
f1.write(papersText)
f1.close()
