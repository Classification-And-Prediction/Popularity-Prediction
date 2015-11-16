import pandas as pd
import numpy as np
import re, nltk        
from nltk.stem.porter import PorterStemmer
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA

train_data_df = pd.read_csv('trainset.csv', header=None, delimiter="\t", quoting=3)
test_data_df = pd.read_csv('popularTest.csv', header=None,delimiter="\t" , quoting=3)

train_data_df.columns = ["Popularity","Title","Tags","Body"]
test_data_df.columns = ["Title","Tags","Body"]

print train_data_df.shape
print test_data_df.shape

print train_data_df.Popularity.value_counts()

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
	text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b"," ",text)
	text = re.sub("\\b[a-zA-Z0-9]{0,1}\\b"," ",text)
	tokens = nltk.word_tokenize(text)
	stems = stem_tokens(tokens, stemmer)
	return stems
	
vectorizer = CountVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 85)
       
X = vectorizer.fit_transform(train_data_df.Text.tolist()).todense()
pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
print len(data2D[:,0])
print len(data2D[:,1])
plt.scatter(data2D[:,0], data2D[:,1], c=train_data_df.Domain)
plt.show()
