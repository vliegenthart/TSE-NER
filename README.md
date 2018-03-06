# TSE-NER

We contribute TSE-NER, an iterative low-cost approach for training NER/NET classifiers for long-tail entity types by exploiting
Term and Sentence Expansion. This approach relies on minimal human input  a seed set of instances of the targeted entity type.
We introduce different strategies for training data extraction, semantic expansion, and result entity filtering.

#Folders
-The "crf_trained_files" contains the trained models
-The "data" folder contains the manually annotated test sets.
-The "evaluation_files" folder contains the files containing the seed terms for the 'dataset' entity type
-The "evaluation_filesMet" folder contains the files containing the seed terms for the 'method' entity type
-The "models" folder contains the word2vec and doc2vec models trained on the corpus
-The "postprocessing" folder contains the scripts for training data generation and filtering
-The "preprocessing" folder contains the scripts for term and sentence expansion strategies and the NER training
-The "prop_files" folder contains the property files used for NER training
-The "stanford_files" folder contains the stanford-ner.jar


#Training a new model
-Change the ROOTPATH in the default_config.py.
-Put your seed terms in the "/data/dataset-names-train.txt"
-Index all the sentences of the publication's text in the corpus using the elasticsearch to extract sentences quickly
-run main.py

#Example
For training an NER model for dataset entity type we need to generate the training data as follows:

we	O
performed	O
a	O
systematic	O
set	O
of	O
experiments	O
using	O
the	O
LETOR	DATA
benchmark	O
collections	O
OHSUMED	DATA
,	O
TD2004	DATA
,	O
and	O
TD2003	DATA

Assume we had a seed term "Letor". Using that term, the "training_data_extraction.extract" function extract all
the sentences that contains that word and that are not in the testing set. However as seen in the above example
also OHSUMED, TD2004 and TD2003 are names of the dataset entity type, but they are not in our seed terms.
In this context, in order to label the other entities also corectly and avoid  the false negatives
we perform the Term Expansion  "expansion.term_expansion_dataset" where we extract all the entities from the text
and we cluster all terms extracted from the sentences with respect to their embedding vectors using K-means, Silhouette
analysis is used to find the optimal number k of clusters. Finally, clusters that contain at least one of the seed
terms are considered to (only) contain entities the same type (e.g Dataset). This means if OHSUMED, TD2004 and TD2003
also appear in the same cluster as Letor they will be label as DATA otherwise O. The entities of the Term expansion step
will be stored in the /evaluation_files folder. Next, we use the seeds and the expanded entities to annotate the sentences
that we extracted using the seeds in the "trainingdata_generation.generate_trainingTE" which should generate the training
data as above. Then we use the training data to traing a new NER. We first generate the property files for the Stanford
NER using the ner_training.create_austenprop. Next we use "ner_training.train" to train the new model. The trained model can
be tested on the test set using "ner_training.test".

For the next iterations, we used the trained-models in the crf_trained_files and use "extract_new_entities.ne_extraction" to
extract all the entities from the corpus. However the extracted entities might be noisy. So we use the filtering techniques to filter
out irrelevant entities using the "filtering". Next the new filtered entities can be used as new seeds for the next iteration...



For the Sentence Expansion approach, everything is the same except, for each sentence (trainingdata_generation.generate_trainingSE), we find its most similar sentence in the corpus
using doc2vec. If the new sentence contains a word of the Term Expansion we will label it accordingly, if not we use it as negative example
