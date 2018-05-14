'''
@author: mesbahs
'''
"""
This script uses the trained NER model in the (crf_trained_files or crf_trained_filesMet folder)
to extract entities from the text of papers and stores them in the post_processing_files folder.
"""
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import stopwords, wordnet
import re
import string
import os
from config import ROOTHPATH, STANFORD_NER_PATH, filter_conference

filterbywordnet = []


def ne_extraction(numberOfSeeds, name, prevnumberOfIteration, numberOfIteration, iteration, es):
    print(f'Started iteration {numberOfIteration-1} with {numberOfSeeds} seeds and expansion type "{name}"...')

    # change crf_trained_files to  crf_trained_filesMet if you want to extract method entities
    path_to_model = ROOTHPATH + '/crf_trained_files/' + name + '_text_iteration' + str(prevnumberOfIteration) + '_splitted' + str(numberOfSeeds) + '_' + str(iteration) + '.ser.gz'

    """
    use the trained Stanford NER model to extract entities from the publications
    """
    nertagger = StanfordNERTagger(path_to_model, ROOTHPATH+STANFORD_NER_PATH)

    newnames = []
    result = []

    for conference in [filter_conference[0]]:

        query = {"query":
            {
                "match": 
                {
                    "journal": conference
                }
            }
        }
        res = es.search(index="ir", doc_type="publications", body=query, size=10000)

        print(str(len(res['hits']['hits'])) + f' documents found for conference {conference}')
        counter = 0
        for doc in res['hits']['hits']:
            counter+=1
            if counter % 20 is 0: print(f'Tagged {counter}/' + str(len(res['hits']['hits'])), 'full texts')
            sentence = doc["_source"]["text"]
            sentence = sentence.replace("@ BULLET", "")
            sentence = sentence.replace("@BULLET", "")
            sentence = sentence.replace(", ", " , ")
            sentence = sentence.replace('(', '')
            sentence = sentence.replace(')', '')
            sentence = sentence.replace('[', '')
            sentence = sentence.replace(']', '')
            sentence = sentence.replace(',', ' ,')
            sentence = sentence.replace('?', ' ?')
            sentence = sentence.replace('..', '.')
            sentence = re.sub(r"(\.)([A-Z])", r"\1 \2", sentence)
            tagged = nertagger.tag(sentence.split())

            for jj, (a, b) in enumerate(tagged):
                # print(jj, a, b)
                # change DATA to MET if you want to extract method entities
                if b == 'DATA':
                    a = a.translate(str.maketrans('', '', string.punctuation))
         

                    try:
                        if tagged[jj + 1][1] == 'DATA':
                            temp = tagged[jj + 1][0].translate(str.maketrans('', '', string.punctuation))

                            bigram = a + ' ' + temp
                            result.append(bigram)
                    except:
                        continue

                    result.append(a)

        # print("Results for", conference , result)

    result = list(set(result))
    result = [w.replace('"', '') for w in result]
    unfiltered_words = result
    filtered_words = [word for word in set(result) if word not in stopwords.words('english')]
    filtered_words = [word for word in filtered_words if not wordnet.synsets(word)]


    for word in set(filtered_words):
        try:
            filterbywordnet.append(word)
            newnames.append(word.lower())
        except:
            newnames.append(word.lower())

    file_path = ROOTHPATH + '/post_processing_files/' + name + '_Iteration' + str(numberOfIteration-1) + str(numberOfSeeds) + '_' + str(iteration) + '.txt'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    f1 = open(file_path, 'w')
    for item in filtered_words:
        f1.write(item + '\n')
    f1.close()

    # TEMP CHANGE - FILTERED & UNFILTERED IS THE SAME
    file_path2 = ROOTHPATH + '/post_processing_files/' + name + '_Iteration' + str(numberOfIteration-1) + str(numberOfSeeds) + '_' + str(iteration) + '_UNFILTERED.txt'
    os.makedirs(os.path.dirname(file_path2), exist_ok=True)

    f1 = open(file_path2, 'w')
    for item in unfiltered_words:
        f1.write(item + '\n')
    f1.close()
