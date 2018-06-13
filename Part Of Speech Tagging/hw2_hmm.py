########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Train a bigram HMM for POS tagging
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from math import log
import numpy as np

# Unknown word token
UNK = 'UNK'

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]

# Class definition for a bigram HMM
class HMM:
### Helper file I/O methods ###
    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                    #print(TaggedWord(token).word)
                sens.append(sentence) # append this list as an element to the list of sentences
            #print(sens)
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                sentence = line.split() # split the line into a list of words
                sens.append(sentence) # append this list as an element to the list of sentences
            #print("sens")
            #print(sens)
            return sens
        else:
            print("Error: unlabeled data file %s ddoes not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script
### End file I/O methods ###

    ################################
    #intput:                       #
    #    unknownWordThreshold: int #
    #output: None                  #
    ################################
    # Constructor
    def __init__(self, unknownWordThreshold=5):
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        ### Initialize the rest of your data structures here ###

    ################################
    #intput:                       #
    #    trainFile: string         #
    #output: None                  #
    ################################
    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        data = self.readLabeledData(trainFile)
        #print(data)# data is a nested list of TaggedWords
        b_tag = {}
        full_b_tag = {}
        self.u_tag = defaultdict(float)
        self.word = defaultdict(float)
        w_tag = defaultdict(float)
        first_tag = defaultdict(float)
        self.transition_prob = {}
        self.emission_prob = {}
        self.initial_prob = {}
        #Creating a count of words dictionary
        for i in range(len(data)):
            for j in range(len(data[i])):
                self.word[data[i][j].word] += 1
        
        for i in range(len(data)):
            for j in range(len(data[i])):               
                self.u_tag[data[i][j].tag] += 1
                if self.word[data[i][j].word] >= self.minFreq:
                    w_tag[data[i][j].word,data[i][j].tag] += 1
                else:
                    w_tag[UNK,data[i][j].tag] += 1
                if j < len(data[i])-1: 
                    tag1 = data[i][j].tag
                    tag2 = data[i][j+1].tag
                    if (tag1,tag2) in b_tag:
                        b_tag[tag1,tag2] += 1
                    else:
                        b_tag[tag1,tag2] = 1
                elif j == len(data[i])-1:
                    if (data[i][j].tag,'<\s>') in b_tag:
                        b_tag[data[i][j].tag,'<\s>'] += 1
                    else:
                        b_tag[data[i][j].tag,'<\s>'] = 1
                if j == 0:
                    first_tag[data[i][j].tag] = first_tag[data[i][j].tag] +1
                    
        for tag1 in self.u_tag:
            for tag2 in self.u_tag:
                if (tag1,tag2) in b_tag:
                    full_b_tag[tag1,tag2] = b_tag[tag1,tag2]
                else:
                    full_b_tag[tag1,tag2] = 0
            if (tag1,'<\s>') in b_tag:
                    full_b_tag[tag1,'<\s>'] = b_tag[tag1,'<\s>']
            else:
                    full_b_tag[tag1,'<\s>'] = 0

        for t1,t2 in full_b_tag:
            self.transition_prob[t1,t2] = (full_b_tag[t1,t2]+1)/(self.u_tag[t1]+ len(self.u_tag)+1)
        
        for word,tag in w_tag:
            self.emission_prob[word,tag] = w_tag[word,tag]/self.u_tag[tag]

        for tag in self.u_tag:
            if tag in first_tag:    
                self.initial_prob[tag] = first_tag[tag]/len(data)
            else:
                self.initial_prob[tag] = 0

        #print(u_tag)
        #print(b_tag)
        #print(w_tag)
        #print(self.emission_prob)     
        #print(self.initial_prob)
        #print(len(data))
        #print(self.transition_prob)
        print("Your first task is to train a bigram HMM tagger from an input file of POS-tagged text")
        #return(transition_prob,emission_prob,initial_prob)
    ################################
    #intput:                       #
    #     testFile: string         #
    #    outFile: string           #
    #output: None                  #
    ################################
    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):
        data = self.readUnlabeledData(testFile)       
        f=open(outFile, 'w+')
        for sen in data:
            vitTags = self.viterbi(sen)
            #print("vitTags")
            #print(vitTags,sen)
            #data1 = self.readUnlabeledData(testFile)
 #       for sen in data1:
            senString = ''
            for i in range(len(sen)):
                #print(i,sen[i])
                senString += sen[i]+"_"+vitTags[i]+" "
            #print(senString)
            print(senString.rstrip(), end="\n", file=f)

    ################################
    #intput:                       #
    #    words: list               #
    #output: list                  #
    ################################
    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        trellis_viterbi = {}
        self.trellis_bpointer = {}
        words_with_UNK = words[:]
        #Convert words not in train data to UNK
        for i in range(len(words_with_UNK)):
            if words_with_UNK[i] not in self.word:
                words_with_UNK[i] = UNK
            elif self.word[words_with_UNK[i]]<self.minFreq :
                words_with_UNK[i] = UNK
                
        #print(words_with_UNK)

        for t in self.u_tag:
            if (words_with_UNK[0],t) in self.emission_prob:
                if self.initial_prob[t] == 0:
                    trellis_viterbi[1,t] = float('-inf')
                else:
                    trellis_viterbi[1,t]= np.log(self.initial_prob[t]) +np.log(self.emission_prob[words_with_UNK[0],t])
                #print(words_with_UNK[0],t)
            else:
                trellis_viterbi[1,t]= float('-inf')
                #print(words_with_UNK[0],t)
            self.trellis_bpointer[1,t]= 0
        for i in range(2,len(words_with_UNK)+1):
            for t2 in self.u_tag:
                if (words_with_UNK[i-1],t2) in self.emission_prob:
                    trellis_viterbi[i,t2] = float('-inf')
                    self.trellis_bpointer[i,t2] = 0
                    for t1 in self.u_tag:
                        tmp = trellis_viterbi[i-1,t1]+ np.log(self.transition_prob[t1,t2]) #Check if this is [t,p] instead
                        if (tmp>trellis_viterbi[i,t2]):
                            trellis_viterbi[i,t2] = tmp
                            self.trellis_bpointer[i,t2] = (list(self.u_tag.keys()).index(t1)+1 )#This starts with 1
                    trellis_viterbi[i,t2] = trellis_viterbi[i,t2]+np.log(self.emission_prob[words_with_UNK[i-1],t2]) 
                else:
                    trellis_viterbi[i,t2] = float('-inf')
                    self.trellis_bpointer[i,t2] = 0
        vit_max = float('-inf')
        t_max = " "

        for t in self.u_tag:
            trellis_viterbi[len(words_with_UNK),t] = trellis_viterbi[len(words_with_UNK),t] +np.log(self.transition_prob[t,'<\s>'])
                    
        for t in self.u_tag:
            if (trellis_viterbi[len(words_with_UNK),t] > vit_max):
                t_max = t
                vit_max = trellis_viterbi[len(words_with_UNK),t]
        #print("Your second task is to implement the Viterbi algorithm for the HMM tagger")
        #print(self.trellis_bpointer)
        #print(trellis_viterbi)
        # returns the list of Viterbi POS tags (strings)
        #return ["NULL"]*len(words) # this returns a dummy list of "NULL", equal in length to words
        return self.unpack(len(words_with_UNK),t_max)
       

    def unpack(self,n,t):
        i = n-1
        tags = []
        while (i >= 0):
            tags.append(t)
            keys = list(self.u_tag.keys())
            t = keys[self.trellis_bpointer[i+1,t]-1]
            i = i-1 
        return list(reversed(tags));
                
                    
        
if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')
    
    