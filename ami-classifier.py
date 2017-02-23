from collections import defaultdict
import nltk
import random
import math
import csv
import os
import sys
import re
import random

from nltk.metrics import scores

from sklearn.naive_bayes import BernoulliNB as Bayes
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn.feature_extraction import DictVectorizer


def digram(dig,words):
    '''checks whether the digram dig is in the list words'''
    if len(words) < 2:
        return False
    tup = dig.lower().split(" ")
    for i in range(len(words)-1):
        if words[i] == tup[0] and words[i+1] == tup[1]:
            return True
    return False

def trigram(trig,words):
    '''checks whether the trigram trig is in the list words'''
    if len(words) < 3:
        return False
    trip = trig.lower().split(" ")
    for i in range(len(words)-2):
        if words[i] == trip[0] and words[i+1] == trip[1] and words[i+2] == trip[2]:
            return True
    return False

def negativePolarityStatement(corpus,i):
    '''computes whether the utterance at the index i has negative polarity'''
    words = corpus[i][6].split(" ")
    words = [word.lower() for word in words]
    if [word for word in words if word in ["cannot", "never", "not", "nothing"]]:
        return True

    for j in range(1,len(words)):
        if words[j].endswith("n't"):
            pos = nltk.pos_tag(words)
            try:
                if pos[j+1][1] == "PRP":
                    continue
            except IndexError:
                return True
            return True

    return False

def answerPolarity(words):
    '''computes the polarity of the utterance that has the surface form of the words in words'''
    reverse = False
    for j in range(len(words[:5])):
        if words[j] == "no" or words[j] == "nope" or words[j] == "nah":
            for k in range(j,min(len(words),j+2)):
                if words[k] in ["but"]:
                    reverse = True
            if not reverse:
                return "neg"
            return "neg"

    reverse = False
    for j in range(len(words[:5])):
        if words[j] == "yes" or words[j] == "yeah" or words[j] == "yep":
            for k in range(j,min(len(words),j+2)):
                if words[k] in ["but"]:
                    reverse = True
            if not reverse:
                return "pos"
            return "pos"

    for j in range(len(words[:5])):
        if words[j].endswith("n't") or words[j] in ["not"]:
            return "neg"

    return "pos"

def containsNegation(corpus,i):
    '''checks whether the utterance at index i contains a negation'''
    words = corpus[i][6].split(" ")
    words = [word.lower() for word in words]
    for j in range(len(words)):
        if words[j].endswith("n't") or words[j] in ["not"]:
            return True
    return False

def isNegatedAux(word,pos):
    '''auxiliary for embedded negation'''
    for i in range(len(pos)-1):
        if pos[i][0] == word and pos[i+1][0] == "not":
            return True
    return False

def isNegatedMain(word,pos):
    '''auxiliary for embedded negation'''
    for i in range(len(pos)-1):
        if pos[i][0] == "not" and pos[i+1][0] == word:
            return True
    return False

def occursAsPrepVerb(prep,verb,pos):
    '''auxiliary for embedded negation'''
    for i in range(len(pos)-1):
        if pos[i] == (prep,"PRP") and pos[i+1][0] == verb and (pos[i+1][1].startswith("V") or verb == "would"):
            return True
    return False

def negatedAuxVerb(pos):
    '''auxiliary for embedded negation'''
    for i in range(len(pos)-2):
        if pos[i][1] == "PRP" and (pos[i+1][1].startswith("V") or pos[i+1][0] == "would") and pos[i+2][0] == "not":
            return pos[i][0], pos[i+1][0]
        if pos[i][0] in ["that","one"] and pos[i+1][0] == "is" and pos[i+2][0] == "not":
            return "it","is"
    return "",""

def negatedAuxVerb2(pos):
    '''auxiliary for embedded negation'''
    for i in range(len(pos)-5):
        if pos[i][0] == "i" and pos[i+1][0] == "do" and pos[i+2][0] == "not" and pos[i+3][0] in ["think","know"] and pos[i+4][1] == "PRP" and pos[i+5][1].startswith("V"):
            return pos[i+4][0],pos[i+5][0]

    for i in range(len(pos)-6):
        if pos[i][0] == "i" and pos[i+1][0] == "do" and pos[i+2][0] == "not" and pos[i+3][0] in ["think","know"] and pos[i+4][0] == "if" and pos[i+5][1] == "PRP" and pos[i+6][1].startswith("V"):
            return pos[i+5][0],pos[i+6][0]

    return "",""

def negatedMainVerb(pos):
    '''auxiliary for embedded negation'''
    for i in range(len(pos)-3):
        if pos[i][1] in ["PRP"] and (pos[i+1][1].startswith("V") or pos[i+1][0] == "would") and pos[i+2][0] == "not" and pos[i+3][1].startswith("V"):
            return pos[i][0], pos[i+3][0]
    return "",""

def normalize(word,stemmer):
    if word == "n't":
        return "not"
    if word == "'s":
        return "is"
    return stemmer.stem(word.lower())

def patternPrepAuxNeg(corpus,i,target):
    '''checks for the pattern preposition -- auxiliary verb -- negation'''
    replyText = nltk.word_tokenize(corpus[i][6])
    targetText = nltk.word_tokenize(corpus[target][6])

    replyPos = nltk.pos_tag(replyText)
    targetPos = nltk.pos_tag(targetText)

    porter = nltk.PorterStemmer()
    replyPos = [(normalize(tup[0],porter),tup[1]) for tup in replyPos if isWord(tup[0])]
    targetPos = [(normalize(tup[0],porter),tup[1]) for tup in targetPos if isWord(tup[0])]

    prep, aux = negatedAuxVerb(targetPos)
    if aux:
        if (occursAsPrepVerb(prep,aux,replyPos[:5])) and (not isNegatedAux(aux,replyPos)):
            return True

    prep, aux = negatedAuxVerb2(targetPos)
    if aux:
        if (occursAsPrepVerb(prep,aux,replyPos[:5])) and (not isNegatedAux(aux,replyPos)):
            return True
    return False

def patternPrepNegVerb(corpus,i,target):
    '''checks for the pattern preposition -- negation -- verb'''
    replyText = nltk.word_tokenize(corpus[i][6])
    targetText = nltk.word_tokenize(corpus[target][6])

    replyPos = nltk.pos_tag(replyText)
    targetPos = nltk.pos_tag(targetText)

    porter = nltk.PorterStemmer()
    replyPos = [(normalize(tup[0],porter),tup[1]) for tup in replyPos if isWord(tup[0])]
    targetPos = [(normalize(tup[0],porter),tup[1]) for tup in targetPos if isWord(tup[0])]

    prep, main = negatedMainVerb(targetPos)
    if main:
        if (occursAsPrepVerb(prep,main,replyPos[:7]) or occursAsPrepVerb("you",main,replyPos[:7])) and (not isNegatedMain(main,replyPos)):
            return True
    return False

def removeStopwords(words):
    stopwords = list(nltk.corpus.stopwords.words('english'))
    newwords = []
    for word in words:
        if not word in stopwords:
            newwords.append(word)
    return newwords

def findTurn(corpus,i):
    '''computes the dialogue turn that the utterance at index i is in.'''
    speaker = corpus[i][5]
    start = i
    end = i+1
    length = 1
    while True:
        try:
            if corpus[end][5] == speaker:
                end += 1
                length += 1
            elif corpus[end][2] in ['bck','fra']:
                end += 1
            else:
                break
        except IndexError:
            break

    while True:
        try:
            if corpus[start-1][5] == speaker:
                start -= 1
                length += 1
            elif corpus[start-1][2] in ['bck','fra']:
                start -= 1
            else:
                break
        except IndexError:
            break

    while True:
        try:
            if not corpus[start][5] == speaker:
                start += 1
            else:
                break
        except IndexError:
            break

    while True:
        try:
            if not corpus[end-1][5] == speaker:
                end -= 1
            else:
                break
        except IndexError:
            break

    return (start,end,length)

def findTarget(corpus,i):
    '''if the utterance at index i is a second-part of an adjacency pair, returns the index of the first part. Otherwise returns i.'''
    if not corpus[i][3] in ['POS','NEG']:
        return i
    if corpus[i][4] == '-':
        return i
    timestamp = corpus[i][4]
    index = corpus[i][0]
    for k in range(i):
        if corpus[k][1] == timestamp and corpus[k][0] == index:
            return k
    return i

def findNext(corpus,i,speaker):
    '''finds first utterance of next turn'''
    for k in range(i+1,len(corpus)):
        if corpus[k][5] == speaker and (not corpus[k][2] in ['bck','fra']):
            return k
    return i

def isWord(word):
    return (re.match("^['\w-]+$",word) is not None)

def findWords(corpus,i,lower=True):
    '''computes the list of words corresponding to the surface form of an utterance'''
    candidates = corpus[i][6].split(" ")
    retVal = []
    for s in candidates:
        if isWord(s):
            if lower:
                retVal.append(s.strip().lower())
            else:
                retVal.append(s.strip())
    return retVal

def features(corpus,i):
    '''Feature selection for the classifier.'''
    features = {}
    words = findWords(corpus,i)

    speaker = corpus[i][5]

    features['len_2'] = len(words) > 2
    features['len_12'] = len(words) > 12 
    features['len_24'] = len(words) > 24

    target = findTarget(corpus,i)
    target_speaker = corpus[target][5]

    words = words[:5]
    
    acceptanceCues = ['yeah', 'absolutely', 'okay', 'accept', 'correct', 'either']
    acues = len([word for word in words if word in acceptanceCues])
    for j in range(len(words)):
        if words[j] in ["true","sure"]:
            if j > 0:
                if words[j-1] != "not":
                    acues += 1
            else:
                acues += 1
    features['a_cues'] = acues 

    rejectionCues = ['but','actually','though','although']#,'problem','difficult','ugly','bad','annoying','lame']
    rcues = len([word for word in words if word in rejectionCues])
    features['r_cues'] = rcues 

    init_hedges = 0
    if words[0] == "well":
        init_hedges += 1
    elif words[0] == "oh":
        init_hedges += 1
    elif words[0] == "uh":
        init_hedges += 1

    features['init_hedges'] = init_hedges

    features['no_no'] = digram("no no",words[:5])
    features['yeah_but'] = digram("yeah but",words[:5])

    cuewords = ['yes', 'but']
    for cue in cuewords: #rejectionCues+acceptanceCues:
        features["f_"+cue] = len([word for word in words[:5] if word == cue])

    features['local-positive'] = answerPolarity(words) == "pos"
    features['local-negative'] = answerPolarity(words) == "neg"
    return features

    if (not negativePolarityStatement(corpus,target)) and answerPolarity(words) == "neg":
        features['positive-negative'] = True
    else:
        features['positive-negative'] = False

    if negativePolarityStatement(corpus,target) and answerPolarity(words) == "neg":
        features['negative-negative'] = True
    else:
        features['negative-negative'] = False

    if negativePolarityStatement(corpus,target) and answerPolarity(words) == "pos":
        features['negative-positive'] = True
    else:
        features['negative-positive'] = False

    if containsNegation(corpus,target):
        features['negative-positive-pattern'] = patternPrepNegVerb(corpus,i,target) or patternPrepAuxNeg(corpus,i,target)
    else:
        features['negative-positive-pattern'] = False

    if (not negativePolarityStatement(corpus,target)) and answerPolarity(words) == "pos":
        features['positive-positive'] = True
    else:
        features['positive-positive'] = False

    return features

def isCandidate(corpus,i):
    '''checks whether the utterance at index i is a candidate for classification. That is, whether it is a positive or negative assessment of an earlier utterance.'''
    utt = corpus[i]
    # b is a boolean
    b = (utt[3] in ['POS','NEG']) and (utt[2] == 'ass') # is either positive or negative assessment
    b = b and findTurn(corpus,i)[0] == i # is first in its turn
    target = findTarget(corpus,i)
    b = b and target < i
    b = b and (corpus[target][2] != 'el.ass') and (corpus[target][2] != "el.inf")
    return  b

def highPrecisionSelection(corpus,i):
    '''unigram 'yeah' are excluded'''
    if findWords(corpus,i) == ['yeah']:
        return True
    return False

def prettyPrint(corpus,start,end,mark):
    '''pretty print for debugging'''
    for i in range(start,end):
        utt = corpus[i]
        if not i == mark:
            print utt[2]+" "+utt[5]+": "+utt[6]
        else:
            print utt[3]+" "+utt[2]+" "+utt[5]+": "+utt[6]

# Script starts here.


obsv = os.listdir('csv')
mistags = [] # to exclude mis-annotated observations
corpus = {}
index = 0
for obs in obsv:
    obsname = obs.split("-")[0]
    if obsname in mistags:
        continue
    corpus[obs] = []
    with open('csv/'+obs) as csvfile:
        reader = csv.reader(csvfile,delimiter="\t",quotechar='"')
        i = 0
        for row in reader:
            i += 1
            if i < 4:
                continue
            row2 = [index]
            for element in row:
                row2.append(element.strip())
            corpus[obs].append(row2)
    index += 1


featuresets = []
utts = []

test = False

i2 = 0

for obs in corpus.keys():
    for i in range(len(corpus[obs])):
        if (i == 0):
            pass
        if isCandidate(corpus[obs],i):
            if highPrecisionSelection(corpus[obs],i):
                continue
            utt = corpus[obs][i]
            feats = features(corpus[obs],i)
            featuresets.append((feats,utt[3]))
            utts.append((obs,i))
            target = findTarget(corpus[obs],i)
            start, end, length = findTurn(corpus[obs],i)
            words = findWords(corpus[obs],i)
            i2 += 1

            if test and feats['negative-negative'] and utt[3] == 'NEG':
                prettyPrint(corpus[obs],target,end,i)
                print "--"
if test:
    sys.exit()

random.seed()
random.shuffle(featuresets)

n = 0
prec = 0
recall = 0
f_measure = 0

refsets = defaultdict(set)
testsets = defaultdict(set)

def label2number(s):
    assert s in ["POS","NEG"]
    if s == "POS":
        return 1
    else:
        return 0

out = False

X = [item[0] for item in featuresets]
Y = [label2number(item[1]) for item in featuresets]
vec = DictVectorizer()
X = list(vec.fit_transform(X).toarray())

cval = 10

for k in range(0,cval):
    i = k*(len(X)/cval)
    test_set_X = X[i:i+len(X)/cval]
    test_set_Y = Y[i:i+len(X)/cval]
    train_set_X = X[:i] + X[i+len(X)/cval:]
    train_set_Y = Y[:i] + Y[i+len(X)/cval:]

    refsets = defaultdict(set)
    testsets = defaultdict(set)

    classifier = Bayes(fit_prior=True)
    classifier.fit(train_set_X,train_set_Y)

    obs = classifier.predict(test_set_X)
    for j in range(len(test_set_X)):
        if test_set_Y[j] == 1:
            refsets['POS'].add(j)
        else:
            refsets['NEG'].add(j)

        if obs[j] == 1:
            testsets['POS'].add(j)
            if False and (test_set_Y[j] == 0):
                obsv = utts[i+j][0]
                index = utts[i+j][1]
                target = findTarget(corpus[obsv],index)
                start, end, length = findTurn(corpus[obsv],index)
                prettyPrint(corpus[obsv],target,end,index)
                print "--"
        else:
            testsets['NEG'].add(j)

    n += 1
    p =  scores.precision(refsets['NEG'],testsets['NEG'])
    r =  scores.recall(refsets['NEG'],testsets['NEG'])
    f =  scores.f_measure(refsets['NEG'],testsets['NEG'])

    try:
        prec += p
        recall += r
        f_measure += f
    except TypeError:
        n -= 1

print n, " validations."
print "Average precision: ", prec/n
print "Average recall: ", recall/n
print "Average F1: ", f_measure/n
