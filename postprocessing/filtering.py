'''
@author: mesbahs
'''
"""
This script will be used to filter the noisy extracted entities.
"""
from numbers import Number
from sklearn.preprocessing import StandardScaler
import codecs, numpy
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
import gensim
from sklearn.cluster import KMeans
from config import ROOTHPATH
import nltk
import requests
from xml.etree import ElementTree
from postprocessing import normalized_pub_distance
import os

class autovivify_list(dict):
    '''Pickleable class to replicate the functionality of collections.defaultdict'''

    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        '''Override addition for numeric types when self is empty'''
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        '''Also provide subtraction method'''
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError


def build_word_vector_matrix(vector_file, propernouns):
    '''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays'''
    numpy_arrays = []
    labels_array = []
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for c, r in enumerate(f):
            sr = r.split()

            try:
                if sr[0] in propernouns and not wordnet.synsets(sr[0]) and sr[0].lower() not in stopwords.words(
                        'english'):
                    labels_array.append(sr[0])
                    # print(sr[0].strip())

                    numpy_arrays.append(numpy.array([float(i) for i in sr[1:]]))
            except:
                continue

    return numpy.array(numpy_arrays), labels_array


def find_word_clusters(labels_array, cluster_labels):
    '''Read the labels array and clusters label and return the set of words in each cluster'''
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(labels_array[c])
    return cluster_to_words

def setup_files(numberOfSeeds, name, numberOfIteration, iteration):
    print("Setting up filtering files...")
    corpus_file_path = ROOTHPATH + '/evaluation_files/X_Seeds_' + str(numberOfSeeds) + '_' + str(iteration) + '.txt'
    pos_file_path = ROOTHPATH + "/evaluation_files/" + name + "_Iteration" + str(numberOfIteration) + "_POS_" + str(numberOfSeeds) + "_" + str(iteration) + ".txt"
    ppf_file_path = ROOTHPATH + '/post_processing_files/' + name + '_Iteration' + str(numberOfIteration) + str(numberOfSeeds) + '_' + str(iteration) + '.txt'

    os.makedirs(os.path.dirname(corpus_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(pos_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(ppf_file_path), exist_ok=True)

    corpus_file = open(corpus_file_path, 'r')
    pos_file = open(pos_file_path, 'w')
    ppf_file = open(ppf_file_path, 'r')

    return [corpus_file, pos_file, ppf_file]

"""
Majority vote filtering
"""

def majorityVote(result):
    finalresult = []
    print(len(result))
    result = list(set(result))
    print(len(result))
    for rr in result:

        count = 0
        # check if in DBpedia
        url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?QueryClass=&QueryString=' + str(rr)
        try:

            resp = requests.request('GET', url)
            root = ElementTree.fromstring(resp.content)
            check_if_exist = []
            for child in root.iter('*'):
                check_if_exist.append(child)
            if len(check_if_exist) == 1:
                count = count + 1
        except:
            pass

        # check if in wordnet or stopword
        if not wordnet.synsets(rr) and rr.lower() not in stopwords.words('english'):
            count = count + 1
        temp = []

        # check PMI
        temp.append(rr)
        temp = normalized_pub_distance.NPD(temp)
        if temp:
            count = count + 1

        if count > 1:
            finalresult.append(rr)
    return finalresult


"""
Embedding clustering filtering
"""

def ec_clustering(numberOfSeeds, name, numberOfIteration, iteration):
    print('started embeding ranking....', numberOfSeeds, name, iteration)
    corpus_file, pos_file, ppf_file = setup_files(numberOfSeeds, name, numberOfIteration, iteration)
    propernouns = []

    # read the extracted entities from the file
    
    with ppf_file as file:
        for row in file.readlines():
            propernouns.append(row.strip())
    dsnames = []
    dsnamestemp = []
    
    with corpus_file as file:
        for row in file.readlines():
            dsnames.append(row.strip())
            propernouns.append(row.strip())

    # read the new seed terms (if exist)
    for i in range(1, int(numberOfIteration) + 1):
        try:
            with open(ROOTHPATH + '/evaluation_files/' + name + '_Iteration' + str(i) + '_POS_' + str(numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
                for row in file.readlines():
                    dsnames.append(row.strip())
                    propernouns.append(row.strip())
        except:
            continue

    newpropernouns = []
    for pp in propernouns:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigrams = list(nltk.bigrams(pp.split()))
            for bi in bigrams:
                aa = bi[0].translate(str.maketrans('', '', string.punctuation))
                bb = bi[1].translate(str.maketrans('', '', string.punctuation))
                bi = aa.lower() + '_' + bb.lower()

                newpropernouns.append(bi)
        else:
            newpropernouns.append(pp)

    dsnames = [x.lower() for x in dsnames]
    dsnames = [s.translate(str.maketrans('', '', string.punctuation)) for s in dsnames]

    sentences_split = [s.lower() for s in newpropernouns]

    df, labels_array = build_word_vector_matrix(ROOTHPATH + "/models/modelword2vecbigram.txt", sentences_split)

    sse = {}
    maxcluster = 0
    if len(df) >= 9:
        for n_clusters in range(2, 10):
            finallist = []

            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_

            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labelss = kmeans_model.fit_predict(df)
            sse[n_clusters] = kmeans_model.inertia_

            for c in cluster_to_words:
                counter = dict()
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in dsnames:

                        for ww in cluster_to_words[c]:
                            finallist.append(ww.replace('_', ' '))

            try:

                silhouette_avg = silhouette_score(df, cluster_labelss)
                print("For n_clusters =", n_clusters,
                      "The average silhouette_score is :", silhouette_avg)
                if silhouette_avg > maxcluster:
                    maxcluster = silhouette_avg
                    
                    finallist = list(set(finallist))

                    for item in finallist:
                        if item.lower() not in dsnamestemp and item.lower() not in dsnames:
                            pos_file.write("%s\n" % item)

            except:
                print("ERROR:::Silhoute score invalid")
                continue
    else:
        for n_clusters in range(2, len(df)):
            finallist = []

            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_

            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labelss = kmeans_model.fit_predict(df)
            sse[n_clusters] = kmeans_model.inertia_

            for c in cluster_to_words:
                print(cluster_to_words[c])

            for c in cluster_to_words:
                counter = dict()
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in dsnames:

                        for ww in cluster_to_words[c]:
                            if ww not in dsnames:
                                finallist.append(ww.replace('_', ' '))
            try:

                silhouette_avg = silhouette_score(df, cluster_labelss)
                print("For n_clusters =", n_clusters,
                      "The average silhouette_score is :", silhouette_avg)
                if silhouette_avg > maxcluster:
                    maxcluster = silhouette_avg
                    
                    finallist = list(set(finallist))

                    for item in finallist:
                        if item.lower() not in dsnamestemp and item.lower() not in dsnames:
                            pos_file.write("%s\n" % item)

            except:
                print("ERROR:::Silhoute score invalid")
                continue


"""
Knowledgebase look-up + EC filtering
"""


def Kb_ecall(numberOfSeeds, name, numberOfIteration, iteration):
    # for iteration in range(0,10):
    propernouns = []
    print('filteriiingg....' + str(numberOfSeeds) + '_' + str(name) + '_' + str(iteration))
    corpus_file, pos_file, ppf_file = setup_files(numberOfSeeds, name, numberOfIteration, iteration)
    
    with ppf_file as file:
        for row in file.readlines():
            propernouns.append(row.strip())

    dsnames = []

    
    with corpus_file as file:
        for row in file.readlines():
            dsnames.append(row.strip())
            propernouns.append(row.strip())
    for i in range(1, int(numberOfIteration)):
        try:
            with open(ROOTHPATH + '/evaluation_files/' + name + '_Iteration' + str(i) + '_POS_' + str(
                    numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
                for row in file.readlines():
                    dsnames.append(row.strip())
                    propernouns.append(row.strip())
        except:
            continue
    newpropernouns = []
    bigrams = []
    for pp in propernouns:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigram = list(nltk.bigrams(pp.split()))

            for bi in bigram:
                bi = bi[0].lower() + '_' + bi[1].lower()
                # print(bi)
                newpropernouns.append(bi)
        else:
            newpropernouns.append(pp)

    dsnames = [x.lower() for x in dsnames]
    dsnamestemp = [s.translate(str.maketrans('', '', string.punctuation)) for s in dsnames]
    finalds = []
    for ds in dsnamestemp:
        dss = ds.split(' ')
        if len(dss) > 1:
            ds = ds.replace(' ', '_')
        finalds.append(ds)

    sentences_split = [s.lower() for s in newpropernouns]

    sentences_split = [s.replace('"', '') for s in sentences_split]

    df, labels_array = build_word_vector_matrix(
        ROOTHPATH + "/models/modelword2vecbigram.txt", sentences_split)

    sse = {}
    maxcluster = 0
    if len(df) >= 9:
        for n_clusters in range(2, 10):
            finallist = []

            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_
            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labelss = kmeans_model.fit_predict(df)
            sse[n_clusters] = kmeans_model.inertia_

            # print("\n")

            for c in cluster_to_words:
                counter = dict()
                dscounter = 0
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in finalds:

                        for ww in cluster_to_words[c]:
                            finallist.append(ww.replace('_', ' '))

            # print(finallist)
            try:

                silhouette_avg = silhouette_score(df, cluster_labelss)

                if silhouette_avg > maxcluster:
                    maxcluster = silhouette_avg
                    
                    finallist = list(set(finallist))

                    for item in finallist:
                        if item.lower() not in dsnamestemp and item.lower() not in dsnames:
                            pos_file.write("%s\n" % item)
                    for item in bigrams:
                        if item.lower() not in dsnamestemp and item.lower() not in dsnames:
                            pos_file.write("%s\n" % item)
                    pos_file.close()
            except:
                print("ERROR:::Silhoute score invalid")
                continue


    else:
        for n_clusters in range(2, len(df)):
            finallist = []

            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_
            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labelss = kmeans_model.fit_predict(df)
            sse[n_clusters] = kmeans_model.inertia_

            # print("\n")

            for c in cluster_to_words:
                counter = dict()
                dscounter = 0
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in finalds:

                        for ww in cluster_to_words[c]:
                            url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?QueryClass=place&QueryString=' + str(
                                ww)
                            resp = requests.request('GET', url)
                            root = ElementTree.fromstring(resp.content)
                            check_if_exist = []
                            for child in root.iter('*'):
                                check_if_exist.append(child)
                            if len(check_if_exist) == 1:
                                finallist.append(ww.replace('_', ' '))

            try:

                silhouette_avg = silhouette_score(df, cluster_labelss)

                if silhouette_avg > maxcluster:
                    maxcluster = silhouette_avg
                    
                    finallist = list(set(finallist))

                    for item in finallist:
                        if item.lower() not in dsnamestemp and item.lower() not in dsnames:
                            pos_file.write("%s\n" % item)
                    for item in bigrams:
                        if item.lower() not in dsnamestemp and item.lower() not in dsnames:
                            pos_file.write("%s\n" % item)
                    pos_file.close()
            except:
                print("ERROR:::Silhoute score invalid")
                continue


"""
Knowledge base look-up filtering
"""


def Kb(numberOfSeeds, name, numberOfIteration, iteration):
    # for iteration in range(0,10):
    propernouns = []
    print('filteriiingg....' + str(numberOfSeeds) + '_' + str(name) + '_' + str(iteration))
    corpus_file, pos_file, ppf_file = setup_files(numberOfSeeds, name, numberOfIteration, iteration)

    with ppf_file as file:
        for row in file.readlines():
            propernouns.append(row.strip())

    dsnames = []
    
    with corpus_file as file:
        for row in file.readlines():
            dsnames.append(row.strip())
            propernouns.append(row.strip())
    for i in range(1, int(numberOfIteration)):
        try:
            with open(ROOTHPATH + '/evaluation_files/' + name + '_Iteration' + str(i) + '_POS_' + str(
                    numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
                for row in file.readlines():
                    dsnames.append(row.strip())
                    propernouns.append(row.strip())
        except:
            continue
    newpropernouns = []

    for pp in propernouns:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigram = list(nltk.bigrams(pp.split()))

            for bi in bigram:
                bi = bi[0].lower() + '_' + bi[1].lower()

                newpropernouns.append(bi)
        else:
            newpropernouns.append(pp)
    finallist = []
    for nn in newpropernouns:
        url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?QueryClass=place&QueryString=' + str(
            nn)
        try:
            resp = requests.request('GET', url)
            root = ElementTree.fromstring(resp.content)
            check_if_exist = []
            for child in root.iter('*'):
                check_if_exist.append(child)
            if len(check_if_exist) == 1:
                finallist.append(nn.replace('_', ' '))
        except:
            finallist.append(nn.replace('_', ' '))
    
    finallist = list(set(finallist))

    for item in finallist:
        if item.lower() not in dsnames:
            pos_file.write("%s\n" % item)
    pos_file.close()


"""
PMI filtering
"""

# corpus_file_path = ROOTHPATH + '/evaluation_files/X_Seeds_' + str(numberOfSeeds) + '_' + str(iteration) + '.txt'
# pos_file_path = ROOTHPATH + "/evaluation_files/" + name + "_Iteration" + str(numberOfIteration) + "_POS_" + str(numberOfSeeds) + "_" + str(iteration) + ".txt"
# ppf_file_path = ROOTHPATH + '/post_processing_files/' + name + '_Iteration' + str(numberOfIteration) + str(numberOfSeeds) + '_' + str(iteration) + '.txt'


def PMI(numberOfSeeds, name, numberOfIteration, iteration):
    # for iteration in range(0,10):
    propernouns = []
    print('Filtering PMI: ' + str(numberOfSeeds) + '_' + str(name) + '_' + str(iteration))
    corpus_file, pos_file, ppf_file = setup_files(numberOfSeeds, name, numberOfIteration, iteration)  

    with ppf_file as file:
        for row in file.readlines():
            propernouns.append(row.strip())
    dsnames = []

    with corpus_file as file:
        for row in file.readlines():
            dsnames.append(row.strip())

    newpropernouns = []
    for pp in propernouns:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigram = list(nltk.bigrams(pp.split()))

            for bi in bigram:
                bi = bi[0].lower() + '_' + bi[1].lower()
                newpropernouns.append(bi)
        else:
            newpropernouns.append(pp)
    finallist = normalized_pub_distance.NPD(newpropernouns)

    
    finallist = list(set(finallist))

    for item in finallist:
        pos_file.write("%s\n" % item)
    pos_file.close()


"""
Majority vote filtering
"""


def MV(numberOfSeeds, name, numberOfIteration, iteration):
    # for iteration in range(0,10):
    propernouns = []
    print('filteriiingg....' + str(numberOfSeeds) + '_' + str(name) + '_' + str(iteration))
    corpus_file, pos_file, ppf_file = setup_files(numberOfSeeds, name, numberOfIteration, iteration)

    """
    Use the extracted entities from the publications
    """
    
    
    with ppf_file as file:
        for row in file.readlines():
            propernouns.append(row.strip())
    dsnames = []
    
    with corpus_file as file:
        for row in file.readlines():
            dsnames.append(row.strip())

    newpropernouns = []
    for pp in propernouns:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigram = list(nltk.bigrams(pp.split()))

            for bi in bigram:
                bi = bi[0].lower() + '_' + bi[1].lower()
                newpropernouns.append(bi)
        else:
            newpropernouns.append(pp)
    finallist = majorityVote(newpropernouns, numberOfSeeds, iteration)

    
    finallist = list(set(finallist))

    for item in finallist:
        pos_file.write("%s\n" % item)
    pos_file.close()
