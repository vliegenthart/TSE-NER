from elasticsearch import Elasticsearch

es = Elasticsearch(
    [{'host': 'localhost', 'port': 9200}])

filter_conference = [ "ESWC", "TPDL"]
for conference in filter_conference:

        query = {"query":
            {"match": {
                "journal": {
                    "query": conference,
                    "operator": "and"
                }
            }
            }
        }

        res = es.search(index="ir", doc_type="publications",
                        body=query, size=10000)
        print(len(res['hits']['hits']))

        for doc in res['hits']['hits']:
            print(doc["_source"]["title"])