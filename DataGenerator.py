import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import pickle

BASE_PHRASES = ['Hi, My name is %s and I am a representative from %s. I am working on updating your system. Can you provide me with your account info to finish this process ?',
'Hi, This is %s from the HR department. We unfortunately have deleted your financial information. This may put some delays on your pay cycle. Can you send us your information ? ',
'Hello, I am Officer %s from the %s Police department. Your name has come up in an investigation on %s. Can you email an account of your whereabouts this past weekend ?',
'Hi, I am %s from %s department. I am unfortunately stuck in %s without any money. Can you run to %s to purchase some gift cards and send me the code on the back so I can purchase a ticket?',
'Congradulations, you have won a trip to %s. Go to the link below to receive your prize.',
'Hi, This is %s from %s department. I saw your good work on the recent project. Could you give me access to your account so I can look at how you designed it ? ',
'Good evening, we have detected strange activity coming from %s on your work account. Go to the link below to update your password.',
'Good morning, I am %s from finance. Can you go to the attached link and update your account info ?',
'Hi all, We are encouraging everyone to update the %s software on your laptop. Go to the link below to update your software',
'Good afternoon, I am %s from %s. We have reason to believe that your system has been compromised. We have fixed things on our end. Go to the link below to update your antivirus software',
'Hi, We are recognizing different employees in the company. You have been chosen for your outstanding work these past few months. We are offering you a trip to %s. Go to the link below to accept.',
'Good morning, we are seeking a few employees to serve as representatives for an upcoming marketing campaign about %s. We think you would be a great fit. Please go to the link below.']


# Class labels/Scores: 0 negative, 1 neutral, 2 positive
SCORES = [1,0,1,0,2,2,0,1,1,0,2,2]
NUM_BLANKS = [s.count('%s') for s in BASE_PHRASES]
STR_LEN = 7

def RandStrings(str_len):
    tmp_s = ''
    a1 = [random.choice(string.ascii_letters) for _ in range(str_len)]
    for q in a1:
        tmp_s = tmp_s + q
    return tmp_s

def DatasetBuilder(num_examples,fname):
    # open  the file
    # looping over the examples ....
    with open(fname,'w') as f:
        for i in range(num_examples):
            # randomly extract an example
            tmp_ex_id = random.choice(range(len(BASE_PHRASES)))
            tmp_ex = BASE_PHRASES[tmp_ex_id]
            # get the number of blank spaces
            tmp_num = NUM_BLANKS[tmp_ex_id]
            # generate random strings
            filler = [RandStrings(STR_LEN) for _ in range(tmp_num)]
            filler = tuple(filler)
            tmp_ex2 = tmp_ex % filler
            tmp_ex2 = tmp_ex2 + '\t'+'{}'.format(SCORES[tmp_ex_id])+'\n'
            f.write(tmp_ex2)
    return 0

def DataExtraction(filename):
    # Note: Assumes the file separates data by tabs
    phrases = []
    scores = []
    with open(filename,'r') as f:
        for line in f:
            data = line.split('\t')
            phrases.append(data[0])
            scores.append(int(data[1]))
    return phrases,np.array(scores)

def TokenizeData(corpus):
    vctzr = TfidfVectorizer()
    vctzr.fit(corpus)
    return vctzr,vctzr.transform(corpus)

def TrainModel(corpus,y):
    vctzr,X = TokenizeData(corpus)
    trX,tsX,trY,tsY = train_test_split(X,y,test_size=0.2,shuffle=True)
    model = RandomForestClassifier()
    model.fit(trX,trY)
    ypred = model.predict(tsX)
    pickle.dump([trX,tsX,trY,tsY,ypred],open('SentimentClsfrTrnAndPreds.pickle','wb'))
    print('Acc. on the test set: {}'.format(accuracy_score(tsY,ypred)))
    return model,vctzr

class EmployeeShell():
    def __init__(self,model,vctzr):
        self.model = model
        self.vctzr = vctzr
        self.sentiments = {0:'negative',1:'neutral',2:'positive'}
    def getMessageSentiment(self,x):
        vect = self.vctzr.transform([x])
        pred = self.model.predict(vect).item()
        return pred #self.sentiments[pred]


class Employee(EmployeeShell):
    def __init__(self,model,vctzr,resp_prob):
        EmployeeShell.__init__(self,model,vctzr)
        self.resp_prob = resp_prob
    def makeDecision(self,x):
        # resp_prob consists of 3 sets of prob depending on the perceived sentiment of the message
        # the decision to follow the instructions or not will either
        # be drawn from a distribution with these probabilities or chosen based on the largest value via argmax
        # ex. resp_prob = [[0.5,0.5],[0.3,0.7],[0.2,0.8]]
        tmp_sentiment = self.getMessageSentiment(x)
        tmp_prob = self.resp_prob[tmp_sentiment]
        action = np.argmax(tmp_prob) #
        # action = np.random.choice([0,1],p=tmp_prob)
        return action
