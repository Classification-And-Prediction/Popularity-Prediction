import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import re, nltk        
from nltk.stem.porter import PorterStemmer
import random
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV

train_data_df = pd.read_csv('train_set.csv', header=None, delimiter="\t", quoting=3)
test_data_df = pd.read_csv('test_set.csv', header=None,delimiter="\n" , quoting=3 ,error_bad_lines=False)

train_data_df.columns = ["Domain","Text"]
test_data_df.columns = ["Text"]

print train_data_df.shape
print test_data_df.shape

print train_data_df.Domain.value_counts()

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub("[http://]", " ", text)
    text = re.sub(" +"," ", text)
	text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b"," ",text)
	text = re.sub("\\b[a-zA-Z0-9]{0,1}\\b"," ",text)
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

vectorizer = TfidfVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 85)

corpus_data_features = vectorizer.fit_transform(train_data_df.Text.tolist() + test_data_df.Text.tolist())
corpus_data_features_nd = corpus_data_features.toarray()

X_train, X_test, y_train, y_test  = train_test_split(body_title_tags_corpus[0:len(train_data_df)], train_data_df.Popularity, random_state=2) 

paramList = []
value = float(10)
for i in range(1,11) :
	paramList.append(value/100)
	value += 10

tuned_parameters = {'C': paramList,'kernel': ('linear','rbf')}

scores = ['precision', 'recall']

for score in scores:

	print("# Tuning hyper-parameters for %s" % score)

	clf = GridSearchCV(LinearSVC(), tuned_parameters, cv=10,scoring='%s' % score)
	clf.fit(body_title_tags_corpus[0:len(train_data_df)],train_data_df.Popularity)
	print("Best parameters set found on development set:")
	print "\n"
	print(clf.best_params_)
	print"\n"
	print("Grid scores on development set:")
	print "\n"
	for params, mean_score, scores in clf.grid_scores_:
		print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
