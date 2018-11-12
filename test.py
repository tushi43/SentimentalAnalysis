import random
import sklearn
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

ip_file = pd.read_csv("Tweets.csv")

reviews = ip_file['text']

id = ip_file['tweet_id']

sentiment = ip_file['airline_sentiment']

rv = {}

#for key, value in reviews.items():
#    print(key, ' ', value)

idx = 1
#URL_REMOVAL
for x in reviews:
    rv[idx] = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',x)
    idx+=1
    #print(x)

#EMOTICONS_REMOVAL
#emoticons_re = re.compile(r'(?<=[\s][\d])(.*?)(?=\\\\[uU])')
#for x in rv:
#    rv[x] = re.sub(emoticons_re,'',rv[x])
#    print(rv[x])

#Usernames_REMOVAL
twitter_username_re = re.compile(r'@([A-Za-z0-9_]+)')
idx = 1
for x in rv.values():
    rv[idx] = re.sub(twitter_username_re,'',x)
    idx+=1
    #print(x)

#TAGS_REMOVAL
twitter_username_re = re.compile(r'#([A-Za-z0-9_]+)')
idx = 1
for x in rv.values():
    rv[idx] = re.sub(twitter_username_re,'',x)
    idx+=1
    #print(rv[x])


#STOP_WORDS_REMOVAL

stop_words = set(stopwords.words('english'))
training_list = {}
#print(stop_words)
#print(rv)

ps = PorterStemmer()
idx = 1
for x in rv.values():
    #print(x)
    word_tokens = word_tokenize(x)
    #ps.stem(w)
    rv[idx]  = [(w) for w in word_tokens if not w in stop_words]
    idx+=1

#print(rv)

#document = [(list(movie_reviews.words(fileid)), category)
             #for category in movie_reviews.categories()
             #for fileid in movie_reviews.fileids(category)]

#print(document)

sent = []

idx=0
for x in sentiment:
    if x == 'positive':
        sent.append('pos')
        idx+=1
    elif x == 'negative':
        sent.append('neg')
        idx+= 1
    else:
        sent.append("neutral")
        idx+=1


#print(idx)
#print(len(rv))
documents = []
#print(sent)

x1=0
id=1

for idx in sent:
    if idx == "pos":
        documents.append((list(rv[id]), sent[id]))
        id+=1
    elif idx == "neg":
        documents.append((list(rv[id]), sent[id]))
        id+=1
    else:
         x1+=1

#print(documents,' ',idx)
#print(movie_reviews.words())movie_reviews.words(fileid)movie_reviews.fileids(category)

all_words = []
for w in rv.values():
    for a in w:
        all_words.append(a.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:1000]


#print(word_features)
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# set that we'll train our classifier with
training_set = featuresets[:5500]

# set that we'll test against.
testing_set = featuresets[5500:]
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set)*100)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set)*100)


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


