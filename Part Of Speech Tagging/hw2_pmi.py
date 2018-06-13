########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
## ## Part 2:
## Use pointwise mutual information to compare words in the movie corpora
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
import numpy as np
import heapq
#----------------------------------------
#  Data input
#----------------------------------------

################################
#intput:                       #
#    f: string                 #
#output: list of list          #
################################
# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file %s ..." % f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            corpus.append(sentence) # append this list as an element to the list of sentences
            #if i % 1000 == 0:
            #    sys.stderr.write("Reading sentence " + str(i) + "\n") # just a status message: str(i) turns the integer i into a string, so that we can concatenate it
        return corpus
    else:
        print("Error: corpus file %s does not exist" % f)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit() # exit the script

#--------------------------------------------------------------
# PMI data structure
#--------------------------------------------------------------
class PMI:
    ################################
    #intput:                       #
    #    corpus: list of list      #
    #output: None                  #
    ################################
    # Given a corpus of sentences, store observations so that PMI can be calculated efficiently
    def __init__(self, corpus):
        print("\nYour task is to add the data structures and implement the methods necessary to efficiently get the pairwise PMI of words from a corpus")
        self.corpus = corpus
        #word_count = defaultdict(float)
        self.sent_count = defaultdict(int) 
        #poss_word_pairs = defaultdict(float)
        self.word_pair_count_sent = {}
        
        for sent in self.corpus:
            words = list(set(sent))
            
            for i in range(len(words)):
                self.sent_count[words[i]] += 1
                for j in range(i+1,len(words)):
                    word_pair = self.pair(words[i],words[j])
                    if word_pair not in self.word_pair_count_sent:
                        self.word_pair_count_sent[word_pair] = 0
                    self.word_pair_count_sent[word_pair] +=1
        #print(self.word_count)
        #print(self.sent_count)
        #print(poss_word_pairs)
        print(len(self.word_pair_count_sent))

    ################################
    #intput:                       #
    #    w1: string                #
    #    w2: string                #
    #output: float                 #
    ################################    
    # Return the pointwise mutual information (based on sentence (co-)occurrence frequency) for w1 and w2
    def getPMI(self, w1, w2):
        #print("\nSubtask 1: calculate the PMI for a pair of words")
        w1,w2 = self.pair(w1,w2)
        if (w1 in self.sent_count) and (w2 in self.sent_count):
            if (w1,w2) in self.word_pair_count_sent:
                prob_w1 = self.sent_count[w1]/len(self.corpus)
                prob_w2 = self.sent_count[w2]/len(self.corpus)
                prob_w1_w2 = self.word_pair_count_sent[w1,w2]/len(self.corpus)
                pmi = np.log2(prob_w1_w2/(prob_w1*prob_w2))
                #print(self.sent_count[w1],self.sent_count[w2],self.word_pair_count_sent[w1,w2])
                return float(pmi) 
            else:
                print("the word pair does not exist in training data")
                return float(0)
        else:
            print("One of the words does not exist in training data")
            return float('-inf')               
    ################################
    #intput:                       #
    #    k: int                    #
    #output: list                  #
    ################################
    # Given a frequency cutoff k, return the list of observed words that appear in at least k sentences
    def getVocabulary(self, k):       
        print("\nSubtask 2: return the list of words where a word is in the list iff it occurs in at least k sentences")
        return [word for word in self.sent_count if self.sent_count[word] >= k]

    ################################
    #intput:                       #
    #    words: list               #
    #    N: int                    #
    #output: list of triples       #
    ################################
    # Given a list of words, return a list of the pairs of words that have the highest PMI
    # (without repeated pairs, and without duplicate pairs (wi, wj) and (wj, wi)).
    # Each entry in the list should be a triple (pmiValue, w1, w2), where pmiValue is the
    # PMI of the pair of words (w1, w2)
    def getPairsWithMaximumPMI(self, words, N):
        print("\nSubtask 3: given a list of words, find the pairs with the greatest PMI")
        a = []
        #b = []
        words.sort()
        for i in range(len(words)):
            #if words[i] in self.sent_count:
                for j in range(i+1, len(words)): 
                    #w1,w2 = self.pair(words[i],words[j])
                    if (words[i],words[j]) in self.word_pair_count_sent: 
                        pmi = self.getPMI(words[i],words[j])
                        heapq.heappush(a,(-pmi,(words[i],words[j])))
                        #if (w1,w2) not in b:
                            #Check for pairs not individual presence of words
                            #b.append((w1,w2))
                            #a.append((self.getPMI(w1,w2),w1,w2)) 
        n_pairs = []
        while len(n_pairs)<N:
             (pmi,(w1,w2)) = heapq.heappop(a)   
             n_pairs.append((-pmi,w1,w2))
        return n_pairs
        #p = heapq.nlargest(N, a,key = itemgetter(0))
        #return p
    ################################
    #intput:                       #
    #    numPairs: int             #
    #    wordPairs: list of triples#
    #    filename: string          #
    #output: None                  #
    ################################
    #-------------------------------------------
    # Provided PMI methods
    #-------------------------------------------
    # Writes the first numPairs entries in the list of wordPairs to a file, along with each pair's PMI
    def writePairsToFile(self, numPairs, wordPairs, filename):
        f=open(filename, 'w+')
        count = 0
        for (pmiValue, wi, wj) in wordPairs:
            if count > numPairs:
                break
            count += 1
            print("%f %s %s" %(pmiValue, wi, wj), end="\n", file=f)

    ################################
    #intput:                       #
    #    w1: string                #
    #    w2: string                #
    #output: tuple                 #
    ################################
    # Helper method: given two words w1 and w2, returns the pair of words in sorted order
    # That is: pair(w1, w2) == pair(w2, w1)
    def pair(self, w1, w2):
        return (min(w1, w2), max(w1, w2))

#------------------------------------------- 
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    corpus = readFileToCorpus('movies.txt')
    pmi = PMI(corpus)
    lv_pmi = pmi.getPMI("vader", "luke")
    print("  PMI of \"luke\" and \"vader\": %f" % lv_pmi)
    numPairs = 100
    #k = 2
    for k in 2, 5, 10, 50, 100, 200:
        commonWords = pmi.getVocabulary(k)    # words must appear in least k sentences
        #print(len(commonWords))
        wordPairsWithGreatestPMI = pmi.getPairsWithMaximumPMI(commonWords, numPairs)
        #print(wordPairsWithGreatestPMI)
        pmi.writePairsToFile(numPairs, wordPairsWithGreatestPMI, "pairs_minFreq=%d.txt" % k)

