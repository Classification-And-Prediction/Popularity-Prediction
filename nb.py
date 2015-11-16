import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import re, nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import random
from sklearn.naive_bayes import *
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import array
import matplotlib.pyplot as plt

train_data_df = pd.read_csv('trainset.csv', header=None, delimiter="\t", quoting=3)
test_data_df = pd.read_csv('popularTest.csv', header=None,delimiter="\t" , quoting=3)

train_data_df.columns = ["Popularity","Title","Tags","Body"]
test_data_df.columns = ["Title","Tags","Body"]

print train_data_df.shape
print test_data_df.shape
print train_data_df.Popularity.value_counts()

#stemmer = SnowballStemmer('english')
#stemmer = WordNetLemmatizer()
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

def tokenize(text):

	text = re.sub("[^a-zA-Z]", " ", text)
	text = re.sub("(http://)", " ", text)
	text = re.sub(" +"," ", text)
	text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b"," ", text)
	text = re.sub("\\b[a-zA-Z0-9]{0,1}\\b"," ", text)
	tokens = nltk.word_tokenize(text)
	stems = stem_tokens(tokens,stemmer)
	
	return stems

vectorizer = TfidfVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 85)
vectorizer1 = CountVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 5)
vectorizer2 = CountVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 3)

corpus_data_features = vectorizer.fit_transform(train_data_df.Body.tolist() + test_data_df.Body.tolist())
corpus_data_features1 = vectorizer1.fit_transform(train_data_df.Title.tolist() + test_data_df.Title.tolist())
corpus_data_features2 = vectorizer2.fit_transform(train_data_df.Tags.tolist() + test_data_df.Tags.tolist())

#colors = [float(100),float(225)]

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

#pipeline = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),])        
X = vectorizer.fit_transform(train_data_df.Body.tolist()).toarray()
pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
plt.xlabel("")
print len(data2D[:,0])
print len(data2D[:,1])
plt.scatter(data2D[:,0], data2D[:,1], c=train_data_df.Popularity,color = ['r','b'])
plt.show()

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
"############"
#print body_title_tags_corpus
"############"
X_train, X_test, y_train, y_test  = train_test_split(body_title_tags_corpus[0:len(train_data_df)], train_data_df.Popularity, random_state=2) 

#print body_title_tags_corpus.shape


value_list = []
accu_list = []
precision_list = []
recall_list = []
f1_list = []

value = float(10)

# get predictions

for i in range(1,11) :
	
	value_list.append(value/10)

	nb_model = MultinomialNB(alpha = value/10 ,fit_prior = True)
	#nb_model = GaussianNB(alpha = value/10,fit_prior = True)
	#nb_model = LinearSVC(penalty = 'l2')
   
	#nb_model = nb_model.fit(X=X_train, y=y_train)

	#y_pred = nb_model.predict(X_test)

	accu_score = cross_val_score(nb_model,body_title_tags_corpus[0:len(train_data_df)],train_data_df.Popularity,cv=10,scoring='accuracy').mean()
	#print "\n"
	#print "accuracy score : ",accu_score  
	accu_list.append(accu_score)

	precision_score = cross_val_score(nb_model,body_title_tags_corpus[0:len(train_data_df)],train_data_df.Popularity,cv=10,scoring='precision').mean()
	#print "\n"
	#print "precision score : ",precision_score
	precision_list.append(precision_score)

	recall_score = cross_val_score(nb_model,body_title_tags_corpus[0:len(train_data_df)],train_data_df.Popularity,cv=10,scoring='recall').mean()
	#print "\n"
	#print "recall score : ",recall_score
	recall_list.append(recall_score)

	f1_score = cross_val_score(nb_model,body_title_tags_corpus[0:len(train_data_df)],train_data_df.Popularity,cv=10,scoring='f1').mean()
	#print "\n"
	#print "f1 score : ",f1_score,"\n"
	f1_list.append(f1_score)
	value -= 1

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

#print value_list
# train classifier
plt.plot(value_list,accu_list,'-y',linewidth = 4,label = "accuracy")
plt.plot(value_list,precision_list,'-r', linewidth = 4, label = "precision")
plt.plot(value_list,recall_list,'-b', linewidth = 4, label = "recall")
plt.plot(value_list,f1_list,'-g', linewidth = 4, label = "f1 score")
#plt.ylim([0.86,0.98])
plt.legend(fontsize = 21)#, weight = 'bold')
plt.xticks(fontsize = 21, fontweight = 'semibold')
plt.yticks(fontsize = 21,fontweight = 'semibold')
#plt.title("Smoothing values alpha V/S Scores",fontsize = 21,fontweight = 'bold')
plt.xlabel("alpha values", fontsize = 21,fontweight = 'semibold')
plt.ylabel("Evaluation Scores", fontsize = 21,fontweight = 'semibold')
plt.show()
