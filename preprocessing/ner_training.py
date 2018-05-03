'''
@author: mesbahs
'''
"""
This script is used to train the NER model
"""
import subprocess
import re
from config import ROOTHPATH


# Testing the results of the trained NER model on the testfile
def test(numberOfSeeds, name, numberOfIteration):
    for iteration in range(0, 2):
        outputfile = open(ROOTHPATH + '/crf_trained_files/temp' + numberOfIteration + name + 'testB.txt', 'a')
        command = 'java -cp ' + ROOTHPATH + '/stanford_files/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ' + ROOTHPATH + '/crf_trained_files/' + name + '_text_iteration' + numberOfIteration + '_splitted' + str(
            numberOfSeeds) + '_' + str(iteration) + '.ser.gz -testFile ' + ROOTHPATH + '/data/testB_dataset.txt'
        p = subprocess.call(command,
                            stdout=outputfile,
                            stderr=subprocess.STDOUT, shell=True)
        outputfile.close()


# test('java -cp /Users/sepidehmesbah/Downloads/stanford-ner-2016-10-31/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier /Users/sepidehmesbah/Downloads/stanford-ner-2016-10-31/seednames_splitted2_1.tsv.ser.gz -testFile /Users/sepidehmesbah/Downloads/ner-crf-master/evaluation/X_testB_50_manually_splitted.tsv')


###############
# Training the Stanfor NER model
def train(numberOfSeeds, name, numberOfIteration):
    for iteration in range(0, 2):
        outputfile = open(ROOTHPATH + '/crf_trained_files/temp' + numberOfIteration + name + 'testA.txt', 'a')

        command = 'java -cp ' + ROOTHPATH + '/stanford_files/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop ' + ROOTHPATH + '/prop_files/austen' + str(
            numberOfSeeds) + '_' + str(iteration) + '.prop'

        p = subprocess.call(command,
                            stdout=outputfile,
                            stderr=subprocess.STDOUT, shell=True)


# Generating the property file  for training the Stanfor NER model
def create_austenprop(numberOfSeeds, name, numberOfIteration):
    for iteration in range(0, 2):
        outputfile = open(ROOTHPATH+'/data/austen.prop', 'r')
        text = outputfile.read()
        print(text)
        modifiedpath = 'trainFile=' + ROOTHPATH + '/evaluation_files/' + name + '_text_iteration' + numberOfIteration + '_splitted' + str(
            numberOfSeeds) + '_' + str(iteration) + '.txt'

        modifiedpathtest = 'testFile=' + ROOTHPATH + '/evaluation_files/' + name + '_text_iteration' + numberOfIteration + 'test_splitted' + str(
            numberOfSeeds) + '_' + str(iteration) + '.txt'
        serializeTo = 'serializeTo=' + ROOTHPATH + '/crf_trained_files/' + name + '_text_iteration' + numberOfIteration + '_splitted' + str(
            numberOfSeeds) + '_' + str(iteration) + '.ser.gz'
        edited = re.sub(r'trainFile.*?txt', modifiedpath, text, flags=re.DOTALL)
        edited = re.sub(r'#testFile.*?txt', modifiedpathtest, edited, flags=re.DOTALL)
        edited = re.sub(r'serializeTo.*?gz', serializeTo, edited, flags=re.DOTALL)
        print(edited)
        text_file = open(ROOTHPATH + '/prop_files/austen' + str(numberOfSeeds) + '_' + str(iteration) + '.prop', 'w')
        text_file.write(edited)
        text_file.close()
