
from pymongo import MongoClient
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import nltk
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
###############################

client = MongoClient('localhost:4321')
db=client.pub
pub = client.pub.publications
es = Elasticsearch(
    [{'host': 'localhost', 'port': 9200}], timeout=30, max_retries=10, retry_on_timeout=True
)
es.cluster.health(wait_for_status='yellow', request_timeout=1)

###############################
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
        "title": "",
        "content": "",
        "journal": "",
        "year":""
    }
    for i, r in enumerate(results):
        # try:
        # list_of_sections = list()
        my_dict['dblpkey'] = r['dblpkey']
        my_dict['title'] = r['title']
        my_dict['journal'] = r['booktitle']
        my_dict['year']=r['year']
        # print(r['content']['abstract'])
        try:
            my_dict['content'] = r['content']['fulltext']
        except:
            my_dict['content'] = ""
            # print(my_dict)
            # sys.exit(1)

        list_of_docs.append(my_dict)

        my_dict = {
            "dblpkey": "",
            "title": "",
            "content": "",
            "journal": "",
            "year": ""
        }

    return list_of_docs


filter_conference = ["WWW", "ICSE", "VLDB", "JCDL", "TREC",  "SIGIR", "ICWSM", "ECDL", "ESWC", "TPDL"]
###############################

# query = {"content.fulltext": {"$exists": "true"}}
list_of_pubs=[]
# total = pub.find(query).count()
# bulksize = 50
# iters = math.ceil(total/bulksize)
# print(total)
for booktitle in filter_conference:
    mongo_string_search = {'$and': [{'booktitle': booktitle}, {'content.fulltext': {'$exists': True}}]}
    list_of_pubs.append(return_chapters(mongo_string_search, db))
for pubs in list_of_pubs:
    actions = []
    for cur in pubs:
        print(cur['dblpkey'])
        print(cur['journal'])


        text = cur["content"]

        actions.append({
                    "_index": "ir",
                    "_type": "publications",
                    "_id" : cur['dblpkey'],
                    "journal":cur['journal'],
                    "year":cur['year'],
                    "_source" : {
                        "text" : text,
                        "title": cur["title"]
                    }
                })
    print(len(actions))


    if len(actions) == 0:
            continue

    res = helpers.bulk(es,actions)
    print(res)

