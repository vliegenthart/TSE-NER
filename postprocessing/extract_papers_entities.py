'''
@author: mesbahs
'''
"""
This script extract entities from papers and writes them to entity sets and overview file
"""

# Enable imports from modules in parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import stopwords, wordnet
import re
import string
import os
from config import ROOTHPATH, STANFORD_NER_PATH, filter_conference
import csv

filterbywordnet = []
model_names = ["DATA", "MET"]
facets = { "DATA": 'dataset', "MET": 'method'}
conference = "acl"
iteration = 1
test_papers = ['conf_acl_PapineniRWZ02']
total_entities = {}
facets_columns = ';'.join(facets)

def main():
    print(f'### Extracting entities for papers in conference: {conference} (iteration model {iteration}) ###')

    for file_name in os.listdir(f'{ROOTHPATH}/data/{conference.lower()}/full_text/'):
        if not file_name.endswith(".txt"): continue

        paper_id = file_name.strip(".txt")

        entities = { "DATA": [], "MET": []}
        for model_name in model_names:
            entities[model_name] = extract_entities(model_name, conference, paper_id, iteration)

        total_entities[paper_id] = entities

    conf_overview = read_overview_csv(conference)

    for paper in conf_overview:
        if paper[0] in total_entities.keys():
            paper[2] = f'{len(total_entities[paper[0]]["DATA"])};{len(total_entities[paper[0]]["MET"])}'

    write_arrays_to_csv(conf_overview, conference, ['paper_id', 'has_pdf', facets_columns, 'number_citations', 'booktitle', 'pdf_url', 'year', 'title', 'type', 'authors'])


def extract_entities(model_name, conference, paper_id, iteration):
    # print(f'Extracting "{model_name}" entities for paper: {paper_id} (iteration model {iteration})...')

    # change crf_trained_files to  crf_trained_filesMet if you want to extract method entities
    if model_name.upper() == "DATA":
        path_to_model = ROOTHPATH + '/crf_trained_files/term_expansion_text_iteration0_splitted25_1.ser.gz'
    else:
        path_to_model = f'{ROOTHPATH}/crf_trained_files/trained_ner_{model_name.upper()}.ser.gz'

    """
    use the trained Stanford NER model to extract entities from the publications
    """
    nertagger = StanfordNERTagger(path_to_model, ROOTHPATH+STANFORD_NER_PATH)

    newnames = []
    result = []

    file_path = f'{ROOTHPATH}/data/{conference.lower()}/full_text/{paper_id}.txt'
    file1 = open(file_path, 'r')
    full_text = file1.read()

    sentence = full_text
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
        tag = model_name.upper()        

        # change DATA to MET if you want to extract method entities
        if b == tag:
            a = a.translate(str.maketrans('', '', string.punctuation))
 
            try:
                if tagged[jj + 1][1] == tag:
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

    print(f'{paper_id} ({model_name}):' , len(filtered_words), filtered_words)

    write_entity_set_file(paper_id, conference, filtered_words, model_name)
    return filtered_words

def write_entity_set_file(paper_id, booktitle, entities, model_name):
  facet = facets[model_name.upper()]

  file_path = f'{ROOTHPATH}/data/{booktitle.lower()}/entity_set/{facet}_{paper_id}_entity_set_0.txt'
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with open(file_path, 'w+') as outputFile:
    for e in entities:
      outputFile.write(f'{e}\n')

def read_overview_csv(booktitle):
  file_path = f'{ROOTHPATH}/data/{booktitle.lower()}/{booktitle.lower()}_papers_overview_total.csv'
  csv_raw = open(file_path, 'r').readlines()
  csv_raw = [line.rstrip('\n').split(',') for line in csv_raw]
  csv_raw.pop(0) # Remove header column
  
  return csv_raw

# Write list of tuples to csv file
def write_arrays_to_csv(array_list, booktitle, column_names):
  file_path = f'{ROOTHPATH}/data/{booktitle.lower()}/{booktitle.lower()}_papers_overview_total.csv'
  os.makedirs(os.path.dirname(file_path), exist_ok=True)

  with open(file_path, 'w+') as outputFile:
    csv_out=csv.writer(outputFile)
    csv_out.writerow(column_names)
    
    for array1 in array_list:
      csv_out.writerow(array1)

if __name__=='__main__':
  main()

