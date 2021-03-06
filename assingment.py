from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>")) # Increase cell width
display(HTML("<style>.rendered_html { font-size: 16px; }</style>")) # Increase font size

# Matplotlib conf
import matplotlib.pyplot as plt
from matplotlib import interactive
%matplotlib inline

# Seaborn conf
import seaborn as sns
sns.set_palette(sns.color_palette("seismic"))

# Needed Libraries
import sys
import pandas
import pandas as pd
import numpy as np
import operator
import string
import re

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.stem import *
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV, ShuffleSplit

# Dataset Loading
training_set = pandas.read_csv("./train.csv", quotechar='"', header=0, sep=",")
test_set = pandas.read_csv("./test.csv", quotechar='"', header=0, sep=",")

# Copy the dataframes to be able to come back to the original version
df_train = training_set.copy()
df_test = test_set.copy()



from collections import Counter
sns.countplot(training_set.target, order=[x for x, count in sorted(Counter(training_set.target).items(), key=lambda x: -x[1])], palette="seismic")
plt.xticks(rotation=90);
dataset = pandas.concat([training_set,test_set], sort=True)

def process_text(raw_text):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_text) 
    words = letters_only.lower().split()
    
    stops = set(stopwords.words("english")) 
    not_stop_words = [w for w in words if not w in stops]
    
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized = [wordnet_lemmatizer.lemmatize(word) for word in not_stop_words]
    
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in lemmatized]
    
    return( " ".join( stemmed ))  

dataset['clean_text'] = dataset['text'].apply(lambda x: process_text(x))

def clean(tweet): 
            
    # Special characters
    tweet = re.sub(r"\x89Û_", "", tweet)
    tweet = re.sub(r"\x89ÛÒ", "", tweet)
    tweet = re.sub(r"\x89ÛÓ", "", tweet)
    tweet = re.sub(r"\x89ÛÏWhen", "When", tweet)
    tweet = re.sub(r"\x89ÛÏ", "", tweet)
    tweet = re.sub(r"China\x89Ûªs", "China's", tweet)
    tweet = re.sub(r"let\x89Ûªs", "let's", tweet)
    tweet = re.sub(r"\x89Û÷", "", tweet)
    tweet = re.sub(r"\x89Ûª", "", tweet)
    tweet = re.sub(r"\x89Û\x9d", "", tweet)
    tweet = re.sub(r"å_", "", tweet)
    tweet = re.sub(r"\x89Û¢", "", tweet)
    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)
    tweet = re.sub(r"fromåÊwounds", "from wounds", tweet)
    tweet = re.sub(r"åÊ", "", tweet)
    tweet = re.sub(r"åÈ", "", tweet)
    tweet = re.sub(r"JapÌ_n", "Japan", tweet)    
    tweet = re.sub(r"Ì©", "e", tweet)
    tweet = re.sub(r"å¨", "", tweet)
    tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)
    tweet = re.sub(r"åÇ", "", tweet)
    tweet = re.sub(r"å£3million", "3 million", tweet)
    tweet = re.sub(r"åÀ", "", tweet)

    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)
        
    # Words with punctuations and special characters
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        tweet = tweet.replace(p, f' {p} ')
        
    return tweet

def process_text(raw_text):
    raw_text = clean(raw_text)
    letters_only = re.sub("[^a-zA-Z]", " ",raw_text) 
    words = letters_only.lower().split()
    
    stops = set(stopwords.words("english"))  
    not_stop_words = [w for w in words if not w in stops]
    
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized = [wordnet_lemmatizer.lemmatize(word) for word in not_stop_words]
    
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in lemmatized]
    
    return( " ".join( stemmed )) 

dataset['clean_text'] = dataset['text'].apply(lambda x: process_text(x))

#Modelling

X_train = dataset[0:len(training_set)][["id", "clean_text", "text"]]
y_train = dataset[0:len(training_set)][["target"]]
X_test = dataset[len(training_set):len(dataset)][["id", "clean_text", "text"]]
y_test = dataset[len(training_set):len(dataset)][["target"]]



from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_train.target.values)
target_labels = le.classes_
encoded_y_train = le.transform(y_train.target.values)

count_vect = CountVectorizer(analyzer = "word") 

train_text_features = count_vect.fit_transform(X_train.clean_text)
test_text_features = count_vect.transform(X_test.clean_text)

def train_and_evaluate_classifier(X, yt, estimator, grid):
    """Train and Evaluate a estimator (defined as input parameter) on the given labeled data using accuracy."""
    
    # Cross validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    classifier = GridSearchCV(estimator=estimator, cv=cv,  param_grid=grid, error_score=0.0, n_jobs = -1, verbose = 5, scoring='f1')
    
    # Train the model over and tune the parameters
    print("Training model")
    classifier.fit(X, yt)

    # CV-score
    print("CV-scores for each grid configuration")
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in sorted(zip(means, stds, classifier.cv_results_['params']), key=lambda x: -x[0]):
        print("Accuracy: %0.3f (+/-%0.03f) for params: %r" % (mean, std * 2, params))
    print()

    return classifier

results_df = pd.DataFrame(columns=['Approach', 'Accuracy'])

nb_text_cls = train_and_evaluate_classifier(train_text_features, encoded_y_train, MultinomialNB(), {})

results_df.loc[len(results_df)] = ['NB Baseline', 0.764]
results_df
