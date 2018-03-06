from pymongo import MongoClient
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import math
import requests
import nltk
import _pickle as cPickle
###############################
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
client = MongoClient("localhost:4321")
pub = client.pub.publications
db=client.pub
#res = requests.get("http://localhost:9200")
es = Elasticsearch([{'host': 'localhost', 'port': 9200}],timeout=30, max_retries=10, retry_on_timeout=True)

es.cluster.health(wait_for_status='yellow', request_timeout=1)


def return_chapters(mongo_string_search, db):
    # mongo_string_search = {"dblpkey": "{}".format(dblkey)}
    results = db.publications.find(mongo_string_search)
    chapters = list()
    chapter_nums = list()
    list_of_docs = list()
    # list_of_abstracts = list()
    merged_chapters = list()
    my_dict = {
        "dblpkey": "",
        "paragraphs": list(),
        "title": ""
    }
    for i, r in enumerate(results):
        # try:
        # list_of_sections = list()
        my_dict['dblpkey'] = r['dblpkey']
        my_dict['title'] = r['title']
        # print(r['content']['abstract'])
        paragraphs = []
        try:

            for chapter in r['content']['chapters']:
                # print(r['dblpkey'])
                if (chapter == {}):
                    continue
                    # remove the filter that removes related works
                    # elif str(chapter['title']).lower() in filter_chapters:
                    # print(chapter['title'])

                # print(chapter['title'])
                for paragraph in chapter['paragraphs']:
                    if paragraph == {}:
                        continue
                    paragraphs.append(paragraph)
            my_dict['paragraphs']=paragraphs

            paragraphs = []
        except:
            continue
            # print(my_dict)
            # sys.exit(1)



        list_of_docs.append(my_dict)
        my_dict = {
            "dblpkey": "",
            "paragraphs": list(),
            "title": ""
        }


    return list_of_docs

list_of_pubs=[]
###############################

filter_conference = ["WWW", "ICSE", "VLDB", "JCDL", "TREC",  "SIGIR", "ICWSM", "ECDL", "ESWC", "TPDL"]
for booktitle in filter_conference:
    mongo_string_search = {'$and': [{'booktitle': booktitle}, {'content.chapters': {'$exists': True}}]}
    list_of_pubs.append(return_chapters(mongo_string_search, db))
###############################

print("Total papers:")
print(len(list_of_pubs))
for pubs in list_of_pubs:
    for paper in pubs:

        actions = []

        cleaned = []

        datasetsent = []
        othersent = []
        for paragraph in paper['paragraphs']:
                 if paragraph == {}:
                        continue
                 lines = (sent_detector.tokenize(paragraph.strip()))
                 if len(lines)< 3:
                     continue




                 for i in range(len(lines)):

                         # print(paragraph[i])
                         words = nltk.word_tokenize(lines[i])
                         lengths = [len(x) for x in words]
                         average = sum(lengths) / len(lengths)
                         if average < 4:
                             continue
                         twosentences=''


                         try:
                             twosentences = lines[i]+ ' ' +lines[i-1]


                         except:
                             twosentences = lines[i] + ' ' + lines[i +1]
                         datasetsent.append(twosentences)




                 #cleaned.append(paragraph)

        for num, parag in enumerate(datasetsent):
                        actions.append({
                            "_index": "twosent",
                            "_type": "twosentnorules",
                            "_id": paper['dblpkey']+str(num),

                            "_source" : {

                                "title" : paper['title'],
                                "content.chapter.sentpositive" : parag,
                                "paper_id":paper['dblpkey']




                            }})
        if len(actions) == 0:
            continue

        res = helpers.bulk(es, actions)
        print(res)

