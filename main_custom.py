'''
@author: mesbahs
'''
"""
This scripts generates training data and trains NER using different number of seeds and approach.
The training data should be extracted using the training_data_extraction.py  using the seed terms (done with different number of seeds 10 times)
and be stored in the "evaluation_files folder".

"""

from preprocessing import ner_training, expansion,training_data_extraction
from postprocessing import trainingdata_generation, extract_new_entities, filtering
from config import ROOTHPATH
from gensim.models import Doc2Vec
from elasticsearch import Elasticsearch

modeldoc2vec = Doc2Vec.load(ROOTHPATH + '/models/doc2vec.model')
es = Elasticsearch(
    [{'host': 'localhost', 'port': 9200}])

# seeds=[5,10,25,50,100]
seeds=[25]


"""
Extract training data for different number of seeds
"""
print("----------------------------")
print("-     EXTRACTING SEEDS     -")
print("----------------------------")

for seed in seeds:
    training_data_extraction.extract(seed)


"""
Term expansion approach for the first iteration
"""
# perform term expansion on the text of the training data using different number of seeds (i.e. 5,10,25,50,100)
# for number in range(0, 10):
    # expansion.term_expansion_dataset(5, 'term_expansion', str(0), str(number))
    # expansion.term_expansion_dataset(10, 'term_expansion', str(0), str(number))
    # expansion.term_expansion_dataset(25, 'term_expansion', str(0), str(number))
    # expansion.term_expansion_dataset(50, 'term_expansion', str(0), str(number))
    # expansion.term_expansion_dataset(100, 'term_expansion', str(0), str(number))

# training data generation
# for number in range(0, 10):
    # trainingdata_generation.generate_trainingTE(5, 'term_expansion', str(0), str(number))
    # trainingdata_generation.generate_trainingTE(10, 'term_expansion', str(0), str(number))
    # trainingdata_generation.generate_trainingTE(25, 'term_expansion', str(0), str(number))
    # trainingdata_generation.generate_trainingTE(50, 'term_expansion', str(0), str(number))
    # trainingdata_generation.generate_trainingTE(100, 'term_expansion', str(0), str(number))

# training the NER model which will be saved in the crf_trained_files folder
# ner_training.create_austenprop(5, 'term_expansion', str(0))
# ner_training.create_austenprop(10, 'term_expansion', str(0))
# ner_training.create_austenprop(25, 'term_expansion', str(0))
# ner_training.create_austenprop(50, 'term_expansion', str(0))
# ner_training.create_austenprop(100, 'term_expansion', str(0))
#
# ner_training.train(5, 'term_expansion', str(0))
# ner_training.train(10, 'term_expansion', str(0))
# ner_training.train(25, 'term_expansion', str(0))
# ner_training.train(50, 'term_expansion', str(0))
# ner_training.train(100, 'term_expansion', str(0))


"""
Sentence approach for the first iteration
"""
# for number in range(0, 10):
    # expansion.term_expansion_dataset(5, 'sentence_expansion', str(0), str(number))
    # expansion.term_expansion_dataset(10, 'sentence_expansion', str(0), str(number))
    # expansion.term_expansion_dataset(25, 'sentence_expansion', str(0), str(number))
    # expansion.term_expansion_dataset(50, 'sentence_expansion', str(0), str(number))
    # expansion.term_expansion_dataset(100, 'sentence_expansion', str(0), str(number))

# for number in range(0, 10):
    # trainingdata_generation.generate_trainingSE(5, 'sentence_expansion', str(0), str(number), modeldoc2vec)
    # trainingdata_generation.generate_trainingSE(10, 'sentence_expansion', str(0), str(number), modeldoc2vec)
    # trainingdata_generation.generate_trainingSE(25, 'sentence_expansion', str(0), str(number), modeldoc2vec)
    # trainingdata_generation.generate_trainingSE(50, 'sentence_expansion', str(0), str(number), modeldoc2vec)
    # trainingdata_generation.generate_trainingSE(100, 'sentence_expansion', str(0), str(number), modeldoc2vec)

# ner_training.create_austenprop(5, 'sentence_expansion', str(0))
# ner_training.create_austenprop(10, 'sentence_expansion', str(0))
# ner_training.create_austenprop(25, 'sentence_expansion', str(0))
# ner_training.create_austenprop(50, 'sentence_expansion', str(0))
# ner_training.create_austenprop(100, 'sentence_expansion', str(0))
#
# ner_training.train(5, 'sentence_expansion', str(0))
# ner_training.train(10, 'sentence_expansion', str(0))
# ner_training.train(25, 'sentence_expansion', str(0))
# ner_training.train(50, 'sentence_expansion', str(0))
# ner_training.train(100, 'sentence_expansion', str(0))

# An example for 5 iterations:
for i in range(0, 1):
    # Extract all entities and write to post_processing_files/
    # extract_new_entities.ne_extraction(25, 'term_expansion', i, i + 1, 0, es)

    # # Filter out entities according to PMI 
    # filtering.PMI(25, 'term_expansion', i, 0)
    
    trainingdata_generation.extract(25, 'term_expansion', i, 0)
    expansion.term_expansion_dataset(25, 'term_expansion', i, 0)
    trainingdata_generation.generate_trainingTE(25, 'term_expansion', i, 0)
    ner_training.create_austenprop(25, 'term_expansion', 0)
    ner_training.train(25, 'term_expansion', 0)

########################################################
