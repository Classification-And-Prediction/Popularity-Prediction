import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import re, nltk        
from nltk.stem.porter import PorterStemmer
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem import WordNetLemmatizer
import random
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import *
from sklearn.svm import *
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import array
import matplotlib.pyplot as plt  
from sklearn.learning_curve import validation_curve

train_data_df = pd.read_csv('trainset.csv', header=None, delimiter="\t", quoting=3)
test_data_df = pd.read_csv('popularTest.csv', header=None,delimiter="\t" , quoting=3)

train_data_df.columns = ["Popularity","Title","Tags","Body"]
test_data_df.columns = ["Title","Tags","Body"]

print train_data_df.shape
print test_data_df.shape
print train_data_df.Popularity.value_counts()

#stemmer = SnowballStemmer('english')
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

def tokenize(text):
    
	text = re.sub("[^a-zA-Z]", " ", text)
	text = re.sub("(http://)", " ", text)
	text = re.sub(" +"," ",text)
	text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b"," ",text)
	text = re.sub("\\b[a-zA-Z0-9]{1,1}\\b"," ",text)
	tokens = nltk.word_tokenize(text)
	stems = stem_tokens(tokens, stemmer)
	return stems

vectorizer = TfidfVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 85)
vectorizer1 = TfidfVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 5)
vectorizer2 = TfidfVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 3)

corpus_data_features = vectorizer.fit_transform(train_data_df.Body.tolist() + test_data_df.Body.tolist())
corpus_data_features1 = vectorizer1.fit_transform(train_data_df.Title.tolist() + test_data_df.Title.tolist())
corpus_data_features2 = vectorizer2.fit_transform(train_data_df.Tags.tolist() + test_data_df.Tags.tolist())

corpus_data_features_nd = corpus_data_features.toarray()
corpus_data_features_nd1 = corpus_data_features1.toarray()
corpus_data_features_nd2 = corpus_data_features2.toarray()

corpus_data_features_nd = corpus_data_features_nd * 0.2
corpus_data_features_nd1 = corpus_data_features_nd1 * 0.5
corpus_data_features_nd2 = corpus_data_features_nd2 * 0.3

body_title_corpus = []
body_title_tags_corpus = []

for i in range(len(corpus_data_features_nd)):

    body_title_corpus.append(np.concatenate((corpus_data_features_nd[i], corpus_data_features_nd1[i])))

body_title_corpus = array(body_title_corpus)

for i in range(len(corpus_data_features_nd)):
    body_title_tags_corpus.append(np.concatenate((body_title_corpus[i], corpus_data_features_nd2[i])))

body_title_tags_corpus = array(body_title_tags_corpus)

print body_title_tags_corpus    

X_train, X_test, y_train, y_test  = train_test_split(body_title_tags_corpus[0:len(train_data_df)], train_data_df.Popularity, random_state=2) 

print body_title_tags_corpus.shape

value_list = []
accu_list = []
precision_list = []
recall_list = []
f1_list = []

value = float(1)
l = [0.01,1.00,100.00]
# get predictions

for i in range(0,3) :
	
	value_list.append(l[i])
	#svm_model = LogisticRegression(penalty = 'l1',dual = False,C = value/10)
	svm_model = LinearSVC(penalty = 'l1',dual = False, C = l[i])
	#svm_model = LinearSVC(penalty = 'l2')

	#svm_model = svm_model.fit(X=X_train, y=y_train)

	#y_pred = svm_model.predict(X_test)

	param_range = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

	X, y = body_title_tags_corpus[0:len(train_data_df)],train_data_df.Popularity
	#param_range = np.logspace(-6, -1, 5)
	train_scores, test_scores = validation_curve(MultinomialNB(fit_prior = False), X, y,param_name="alpha",param_range=param_range,cv=10)

	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.title("Validation Curve with NB")
	plt.xlabel("$\gamma$")
	plt.ylabel("Score")
	plt.ylim(0.0, 1.1)
	plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
	plt.fill_between(param_range, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
	plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
	             color="g")
	plt.fill_between(param_range, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
	plt.legend(loc="best")
	plt.show()
	
	accu_score = cross_val_score(svm_model,body_title_tags_corpus[0:len(train_data_df)],train_data_df.Popularity,cv=10,scoring='accuracy').mean()
	print "\n"
	print "accuracy score : ",accu_score  
	accu_list.append(accu_score)

	precision_score = cross_val_score(svm_model,body_title_tags_corpus[0:len(train_data_df)],train_data_df.Popularity,cv=10,scoring='precision').mean()
	print "\n"
	print "precision score : ",precision_score
	precision_list.append(precision_score)

	recall_score = cross_val_score(svm_model,body_title_tags_corpus[0:len(train_data_df)],train_data_df.Popularity,cv=10,scoring='recall').mean()
	print "\n"
	print "recall score : ",recall_score
	recall_list.append(recall_score)

	f1_score = cross_val_score(svm_model,body_title_tags_corpus[0:len(train_data_df)],train_data_df.Popularity,cv=10,scoring='f1').mean()
	print "\n"
	print "f1 score : ",f1_score,"\n"
	f1_list.append(f1_score)

	value += 1


x = 0
for i in range(len(accu_list)) :
	x += accu_list[i]
print x/len(accu_list)

x = 0
for i in range(len(accu_list)) :
	x += precision_list[i]
print x/len(accu_list)

x = 0
for i in range(len(accu_list)) :
	x += recall_list[i]
print x/len(accu_list)

x = 0
for i in range(len(accu_list)) :
	x += f1_list[i]
print x/len(accu_list)

# train classifier
plt.plot(value_list,accu_list,'-y',linewidth = 4,label = "accuracy")
plt.plot(value_list,precision_list,'-r', linewidth = 4, label = "precision")
plt.plot(value_list,recall_list,'-b', linewidth = 4, label = "recall")
plt.plot(value_list,f1_list,'-g', linewidth = 4, label = "f1 score")
#plt.ylim([0.78,1.00])
plt.legend(fontsize = 21)#, weight = 'bold')
plt.xticks(fontsize = 21, fontweight = 'semibold')
plt.yticks(fontsize = 21,fontweight = 'semibold')
#plt.title("Lambda values for L1 V/S Scores", fontsize = 21,fontweight = 'bold')
plt.xlabel("C values", fontsize = 21,fontweight = 'semibold')
plt.ylabel("Evaluation Scores", fontsize = 21,fontweight = 'semibold')
plt.show()

svm_model = LinearSVC(penalty = 'l1',dual = False, C = 0.6)
#svm_model = LinearSVC(penalty = 'l2')
#svm_model = LinearSVC()
svm_model = svm_model.fit(X=body_title_tags_corpus[0:len(train_data_df)], y=train_data_df.Popularity)
test_pred = svm_model.predict(body_title_tags_corpus[len(train_data_df):])
# sample some of them

spl = random.sample(xrange(len(test_pred)),10)
    
for text, Domain in zip(test_data_df.Title[spl], test_pred[spl]):
    print Domain,"====", text,"\n"
