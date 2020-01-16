import argparse
import re
import statistics as stats
# import stop_words
#import saver
import json
import pickle
import csv
import gensim.models as gm
import random
from collections import Counter
from string import punctuation
import features

# from features import get_embeddings
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression

'''infusing data from different datasets requires different opening file functions:
input: dataset files
output: two lists, for comments and labels respectively'''

def read_not_hate(corpus_file, set_size):
	#reads not hate from DS data. Different from read_hate to call difference ratios of both sets(see line 144)
    t = 0
    comments, labels = [], []
    with open(corpus_file, "r", encoding ='ISO-8859-1') as inputfile:
        for row in csv.reader(inputfile):
            t += 1
            comments.append(row[0])
            labels.append(row[1])
            if t == set_size:
                break
        return comments, labels


def read_hate(corpus_file, set_size):
    t = 0
    comments, labels = [], []
    with open(corpus_file, "r", encoding ='ISO-8859-1') as inputfile:
        for row in csv.reader(inputfile):
            t += 1
            comments.append(row[0])
            labels.append(row[1])
            if t == set_size:
                break
        return comments, labels

def read_test(corpus_file):
	#read test reads the last 1200 rows of evalita
    t = 0
    comments, labels = [], []
    with open(corpus_file, "r", encoding ='ISO-8859-1') as inputfile:
        for row in csv.reader(inputfile):
            if t == 0:
                t += 1
                continue
            labels.append(convert(row[2]))
            comments.append(row[1])
    return  comments, labels

def read_portion_gold(corpus_file, unit): 
	#read_portion_gold reads EVALITA data
    t = 0
    comments, labels = [], []
    with open(corpus_file, "r", encoding ='ISO-8859-1') as inputfile:
        for row in csv.reader(inputfile):
            if t == 0:
                t += 1
                continue
            labels.append(convert(row[2]))
            comments.append(row[1])
            t += 1
            if t == unit:
                print('The current amount of gold data is:{}'.format(unit))
                break
    return  comments, labels

def convert(item):
    if item == "1":
        return "yes"
    else:
        return "no"  

def read_t(corpus_file):
	#read-t reads data from TORINO
    t = 0
    documents, labels = [], []
    with open(corpus_file, "r", encoding ='ISO-8859-1') as inputfile:
        for row in csv.reader(inputfile):
            if t == 0:
                t += 1
                continue
            labels.append(row[3])
            documents.append(row[9])
        return documents, labels

def load_embeddings(embedding_file):
    '''
    loading embeddings from file
    input: embeddings stored as json (json), pickle (pickle or p) or gensim model (bin)
    output: embeddings in a dict-like structure available for look-up, vocab covered by the embeddings as a set
    '''
    if embedding_file.endswith('json'):
        f = open(embedding_file, 'r', encoding='utf-8')
        embeds = json.load(f)
        f.close
        vocab = {k for k,v in embeds.items()}
    elif embedding_file.endswith('bin'):
        embeds = gm.KeyedVectors.load(embedding_file).wv
        vocab = {word for word in embeds.index2word}
    elif embedding_file.endswith('p') or embedding_file.endswith('pickle'):
        f = open(embedding_file,'rb')
        embeds = pickle.load(f)
        f.close
        vocab = {k for k,v in embeds.items()}
    elif embedding_file.endswith('txt'):
        embeds = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
        vocab = embeds.wv.vocab

    return embeds, vocab


if __name__ == '__main__':
    sizes = [0]
    #gold_sizes = [1201, 2401, 3601,4801]
    gold_sizes = [4801]

    for set_size in sizes:
        for gold_size in gold_sizes:
            print('-'*20)
            print('defining the set of %s and %s gold' % (set_size, gold_size))
            #c, l = read_not_hate('hate0.csv', set_size*0.54)
            #ch, lh = read_hate('hate1.csv', set_size*0.46)
            #c,l = read_t('tweet_fixed.csv')
            docs, labels = read_portion_gold('mix.csv', gold_size)

            X_no_hate =  docs
            Y_hate = labels
            
            data = list(zip(X_no_hate, Y_hate))
        
            random.shuffle(data)
        
            X, Y = zip(*data)
        
            print('defining the test set.....')
            documents, labels = read_test('1200.csv')
            X_test = documents
            y_test = labels
        
            # initialize classifier
            tfidf_word = TfidfVectorizer(ngram_range=(1, 3))
            tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
            extra_vec = Pipeline([('extra', ExtraFeatures()), ('vec', DictVectorizer())])
        
            # getting embeddings
            path_to_embs = 'modelNotHate.bin'
            print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
            embeddings, vocab = load_embeddings(path_to_embs)
            print('Done')

        
            pipeline = Pipeline([
                ('features', FeatureUnion([
                            ('tfidf_char', tfidf_char),
							('tfidf', tfidf_word),
							('word_embeds', f.Embeddings(embeddings, pool='pool'))
                    ])),
                ('classifier', LinearSVC())
            ])
        
            #other classifiers
            '''('classifier', svm.SVC(kernel='linear', C=0.5, class_weight = 'balanced')
            ('classifier', svm.SVC(kernel='linear', C=0.5, class_weight = 'balanced')
            LogisticRegression(warm_start=True)'''
        
            pipeline.fit(X, Y)
            y_pred = pipeline.predict(X_test)
        
            print(classification_report(y_test, y_pred))
            print(confusion_matrix(y_test, y_pred))
            print()
            print("F1 score (micro) = {}".format(f1_score(y_test, y_pred, average="micro")))
            print("F1 score (macro) = {}".format(f1_score(y_test, y_pred, average="macro")))
        