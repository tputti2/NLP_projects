########################################
## CS447 Natural Language Processing  ##
##           Homework 1               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Develop a smoothed n-gram language model and evaluate it on a corpus
##
import os.path
import sys
import random
from operator import itemgetter
from collections import defaultdict
import math
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef
#cwd = os.getcwd()
#print(cwd)
#os.chdir('Desktop\\Acads\\NLP\\HW 1')
#my_corpus = readFileToCorpus('train.txt')
#print(my_corpus)

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


# Preprocess the corpus to help avoid sess the corpus to help avoid sparsity
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if freqDict[word] < 2:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

#preprocessed_corpus = preprocess(my_corpus)
#print(preprocessed_corpus)

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        print("""Your task is to implement three kinds of n-gram language models:
      a) an (unsmoothed) unigram model (UnigramModel)
      b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
      c) an unsmoothed bigram model (BigramModel)
      d) a bigram model smoothed using absolute discounting (SmoothedBigramModel)
      """)
    #enddef

    # Generate a sentence by drawing words according to the 
    # model's probability distribution
    # Note: think about how to set the length of the sentence 
    #in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    #emddef

    # Given a sentence (sen), return the probability of 
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    #enddef

    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        return 0.0
    #enddef

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        stringGenerated  = []
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            #stringGenerated = stringGenerated.append(str(prob))
            stringGenerated = sen
            print((prob,stringGenerated), end="\n", file=filePointer)
            #print >> file, prob, " ", sen
	#endfor
    #enddef
#endclass
    
# Unigram language model
class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        print("Subtask: implement the unsmoothed unigram language model")
        self.unigramdist = UnigramDist(corpus)
    #endddef
    
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        word_pick = "<s>"
        unigram_sent = []
        #i = 0
        while(word_pick != "</s>"):
            unigram_sent.append(word_pick)
            word_pick = self.unigramdist.draw()
            #i = i+1
        
        unigram_sent.append("</s>")   
        return unigram_sent
    #enddef 
    
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        prob = 1.0
        #print(sen)
        for index in range(1,len(sen)):
            prob = prob*(self.unigramdist.prob(sen[index]))
            #print(prob,sen[index])
            #print(prob,self.unigramdist.prob(sen[index]),sen[index])
        return prob
    #enddef

    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        N = 0
        p = 0
        for sen in corpus:
            for word in sen:
                if word != "<s>":
                    if (self.unigramdist.prob(word) == 0):  
                        return math.inf
                    N = N+1
                    p = p+ math.log(self.unigramdist.prob(word))
        Perp = math.exp(-p/N)
        return Perp

#endclass

#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        print("Subtask: implement the smoothed unigram language model")
        self.smoothedunigram = Smoothed_UnigramDist(corpus)
    #endddef
    
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        word_pick = "<s>"
        unigram_sent = []
        #i = 0
        while(word_pick != "</s>"):
            unigram_sent.append(word_pick)
            word_pick = self.smoothedunigram.draw()
            #i = i+1
        
        unigram_sent.append("</s>")    
        return unigram_sent
        
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        prob = 1.0
        #print(sen)
        for index in range(1,len(sen)):
            prob = prob*(self.smoothedunigram.prob(sen[index]))
            #print(prob,sen[index])
            #print(prob,self.unigramdist.prob(sen[index]),sen[index])
        return prob
    #enddef    
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        N = 0
        p = 0
        for sen in corpus:
            for word in sen:
                if word != "<s>":
                    if (self.smoothedunigram.prob(word) == 0):  
                        return math.inf
                    N = N+1
                    p = p+ math.log(self.smoothedunigram.prob(word))
        Perp = math.exp(-p/N)
        return Perp    

#endclass

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def __init__(self, corpus):
        print("Subtask: implement the unsmoothed bigram language model")
        self.bigramdist = BigramDist(corpus)
    #endddef
    
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        word_pick = "<s>"
        bigram_sent = []
        #i = 0
        while(word_pick != "</s>"):
            bigram_sent.append(word_pick)
            word_pick = self.bigramdist.draw(word_pick)
            #i = i+1
        
        bigram_sent.append("</s>")    
        return bigram_sent
       
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        #word_list = sen.split()
        #print(sen)
        prob = 1
        for index in range(len(sen)-1):
            word1 = sen[index] 
            word2 = sen[index + 1]
            prob = prob*(self.bigramdist.prob(word1,word2))
        return prob
    #enddef

    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        N = 0
        p = 0
        for sen in corpus:
            for index, word in enumerate(sen):
                if index < len(sen) - 1:
                    word1 = sen[index] 
                    word2 = sen[index + 1]
                    N = N+1
                    if (self.bigramdist.prob(word1,word2) == 0): 
                        return math.inf
                    else:
                        p = p+math.log(self.bigramdist.prob(word1,word2))
        Perp = math.exp(-p/N)
        return Perp                                        
        #for word in sen:
                #if word != "<s>":
                    #if (self.smoothedunigram.prob(word) == 0):  
                        #return math.inf
#endclass

# Smoothed bigram language model (use absolute discounting for smoothing)
class SmoothedBigramModelAD(LanguageModel):
    def __init__(self, corpus):
        print("Subtask: implement the smoothed bigram language model with absolute discounting")
        self.adsmooth_bigramdist = ADSmooth_BigramDist(corpus)
    #endddef
    
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        word_pick = "<s>"
        bigram_sent = []
        i = 0
        while(i<6):
            bigram_sent.append(word_pick)
            word_pick = self.adsmooth_bigramdist.draw(word_pick)
            i = i+1
        
        bigram_sent.append("</s>")    
        return bigram_sent
       
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        #word_list = sen.split()
        prob = 1
        for index in range(len(sen)-1):
            word1 = sen[index] 
            word2 = sen[index + 1]
            prob = prob*(self.adsmooth_bigramdist.prob(word1,word2))
            #print(prob,(word1,word2))    
        return prob
    #enddef
    
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        N = 0
        p = 0
        for sen in corpus:
            for index, word in enumerate(sen):
                if index < len(sen) - 1:
                    word1 = sen[index] 
                    word2 = sen[index + 1]
                    N = N+1
                    p = p+math.log(self.adsmooth_bigramdist.prob(word1,word2))
        Perp = math.exp(-p/N)
        return Perp                                        
        #for word in sen:
                #if word != "<s>":
                    #if (self.smoothedunigram.prob(word) == 0):  
                        #return math.inf    
#endclass

# Smoothed bigram language model (use absolute discounting and kneser-ney for smoothing)
class SmoothedBigramModelKN(LanguageModel):
    def __init__(self, corpus):
        print("Subtask: implement the smoothed bigram language model with kneser-ney smoothing")
        self.knsmooth_bigramdist = KNSmooth_BigramDist(corpus)
    #endddef
    
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        word_pick = "<s>"
        bigram_sent = []
        i = 0
        while(i<6):
            bigram_sent.append(word_pick)
            word_pick = self.knsmooth_bigramdist.draw(word_pick)
            i = i+1
        
        bigram_sent.append("</s>")    
        return bigram_sent
       
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        #word_list = sen.split()
        prob = 1
        for index in range(len(sen)-1):
            word1 = sen[index] 
            word2 = sen[index + 1]
            prob = prob*(self.knsmooth_bigramdist.prob(word1,word2))
        return prob
    #enddef

    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        N = 0
        p = 0
        for sen in corpus:
            for index, word in enumerate(sen):
                if index < len(sen) - 1:
                    word1 = sen[index] 
                    word2 = sen[index + 1]
                    N = N+1
                    p = p+math.log(self.knsmooth_bigramdist.prob(word1,word2))
        Perp = math.exp(-p/N)
        return Perp                                        
        #for word in sen:
                #if word != "<s>":
                    #if (self.smoothedunigram.prob(word) == 0):  
                        #return math.inf        
#endclass

# Sample class for a unsmoothed unigram probability distribution
# Note: 
#       Feel free to use/re-use/modify this class as necessary for your 
#       own code (e.g. converting to log probabilities after training). 
#       This class is intended to help you get started
#       with your implementation of the language models above.
class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if (word != "<s>"):
                    self.counts[word] += 1.0
                    self.total += 1.0
            #endfor
        #endfor
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word):
        #print(self.counts[word])
        return (self.counts[word])/(self.total)
    #enddef

    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word
	    #endif
	#endfor
    #enddef
#endclass

class BigramDist:
    def __init__(self, corpus):
        self.tot_sen = 0.0
        self.unigram = defaultdict(float)
        self.train(corpus)       
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        self.bigrams = {}
        for sen in corpus:
            self.tot_sen = self.tot_sen+1
            for index, word in enumerate(sen):
                if index < len(sen) - 1:
                    word1 = sen[index] 
                    word2 = sen[index + 1]
                    bigram = (word1, word2)
        
                    if bigram in self.bigrams:
                        self.bigrams[bigram] = self.bigrams[bigram] + 1
                    else:
                        self.bigrams[bigram] = 1
        #sorted_bigrams = sorted(self.bigrams.items(), key = lambda pair:pair[1], reverse = True)
    
        #for bigram, count in sorted_bigrams:
            #print(bigram, ":", count)
            
            #for sen in corpus:
                #for i in range(len(sen)-1):
                    #self.bigrams[(sen[i],sen[i+1])] += 1.0                
        for sen in corpus:
            for word in sen:
                if word != "<s>":
                    self.unigram[word] += 1.0
        #endfor
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word1,word2):
        if (word1,word2) in self.bigrams.keys():
            if word1 == "<s>":
                return self.bigrams[(word1,word2)]/self.tot_sen
            else:
                return self.bigrams[(word1,word2)]/self.unigram[word1]
        else: 
            return 0
                                    
    #enddef

    # Generate a single random word according to the distribution
    def draw(self,word1 = "<s>"):
        rand = random.random()
        for (w1,w2) in self.bigrams.keys():
            if (w1 == word1):
                rand -= self.prob(w1,w2)
                if rand <= 0.0:
                    return w2
	    #endif
	#endfor
    #enddef
#endclass

class Smoothed_UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if (word != "<s>"):
                    self.counts[word] += 1.0
                    self.total += 1.0
            #endfor
        #endfor
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word):
        return (self.counts[word]+1)/(self.total+len(self.counts))
    #enddef

    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word
	    #endif
	#endfor
    #enddef
#endclass

class ADSmooth_BigramDist:
    def __init__(self, corpus):
        self.disc_factor = 0
        self.unigram = defaultdict(float)
        self.smoothed_unigram = Smoothed_UnigramDist(corpus)       
        self.train(corpus)
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        self.bigrams = {}
        vocab_initial = []
        for sen in corpus:
            for index, word in enumerate(sen):
                if index < len(sen)-1:
                    self.unigram[word] += 1.0 #This has start token also
                    vocab_initial.append(word) #Vocab has start token
                    word1 = sen[index] 
                    word2 = sen[index+1]
                    bigram = (word1, word2)
                    
                    
                    if bigram in self.bigrams:
                        self.bigrams[bigram] = self.bigrams[bigram] + 1
                    else:
                        self.bigrams[bigram] = 1
                    #Totake care of last word
                if index == len(sen):
                    self.unigram[word] += 1.0
                    vocab_initial.append(word)
                    
        vocab = set(vocab_initial)
        print(len(vocab))

    #def s_value(self,word1):
       # w2list = []
        #for (w1,w2) in self.bigrams.keys():
            #if (w1 == word1):
                #w2list.append(w2)
        #w2_uniquelist = set(w2list)
        
        #This includes bigrams that are possible but not seen in the corpus
        self.final_bigrams = {} 
        self.s_value = {}  
        w2_uniquelist = {}  
        n1 = 0
        n2 = 0
        for word1 in vocab:
            w2_uniquelist[word1] = set([])
            for word2 in vocab:
                pos_bigram = (word1,word2)
                if pos_bigram in self.bigrams.keys():
                    self.final_bigrams[pos_bigram] = self.bigrams[pos_bigram]
                    w2_uniquelist[word1].add(word2) 
                    if (self.final_bigrams[pos_bigram] == 1):
                        n1 = n1+ 1
                    elif (self.final_bigrams[pos_bigram] == 2):
                        n2 = n2+ 1        
                else:
                    self.final_bigrams[pos_bigram] = 0
            self.s_value[word1] = len(w2_uniquelist[word1])
        self.disc_factor = n1/(n1+2*n2)  
        print(len(self.final_bigrams))
                                           
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word1,word2):
        if (word2 != "<s>"):
            max_arg = (self.final_bigrams[(word1,word2)]-self.disc_factor)
            #prob =  (max(max_arg,0)/self.unigram[word1])+ ((self.disc_factor()/self.unigram[word1])*self.s_value(word1)*self.smoothed_unigram.prob(word2))
            num = (self.disc_factor*self.s_value[word1]*self.smoothed_unigram.prob(word2))
            prob =  (max(max_arg,0)+num)/self.unigram[word1]
            #print(max(max_arg,0),word1,word2)
            #print(self.unigram[word1])
            #print(self.disc_factor)
            #print(self.s_value(word1))
            #print(self.smoothed_unigram.prob(word2))
            
            return prob
        else:
            return 0
        #enddef
   
        
    # Generate a single random word according to the distribution
    def draw(self,word1 = "<s>"):
        rand = random.random()
        for (w1,w2) in self.bigrams.keys():
            if (w1 == word1):
                rand -= self.prob(w1,w2)
                if rand <= 0.0:
                    return w2
	    #endif
	#endfor
    #enddef
#endclass

class KNSmooth_BigramDist:
    def __init__(self, corpus):
        self.unigram = defaultdict(float)
        self.train(corpus)
        self.disc_factor
        
        
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        self.bigrams = {}
        for sen in corpus:
            for index, word in enumerate(sen):
                if index < len(sen) - 1:
                    word1 = sen[index] 
                    word2 = sen[index + 1]
                    bigram = (word1, word2)
        
                    if bigram in self.bigrams:
                        self.bigrams[bigram] = self.bigrams[bigram] + 1
                    else:
                        self.bigrams[bigram] = 1
        #sorted_bigrams = sorted(self.bigrams.items(), key = lambda pair:pair[1], reverse = True)
    
        #for bigram, count in sorted_bigrams:
            #print(bigram, ":", count)
            
            #for sen in corpus:
                #for i in range(len(sen)-1):
            #self.bigrams[(sen[i],sen[i+1])] += 1.0                
        vocab_initial = []
        for sen in corpus:
            for index,word in enumerate(sen):
                vocab_initial.append(word)
        vocab = set(vocab_initial)  
        
        self.final_bigrams = {}
        
        for word1 in vocab:
            for word2 in vocab:
                pos_bigram = (word1,word2)
                if pos_bigram in self.bigrams.keys():
                    self.final_bigrams[pos_bigram] = self.bigrams[pos_bigram]
                else:
                    self.final_bigrams[pos_bigram] = 0
                                                
        for sen in corpus:
            for word in sen:
                self.unigram[word] += 1.0
           #Calculate discounting factor
    
        n1 = 0
        n2 = 0
        for (w1,w2) in self.bigrams.keys(): 
           if (self.bigrams[(w1,w2)] == 1):
               n1 = n1+ 1
           elif (self.bigrams[(w1,w2)] == 2):
               n2 = n2+ 1
        self.disc_factor = n1/(n1+2*n2)
        #print("dist_factor is")
        #print(dist)
        return self.disc_factor

        #endfor
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word1,word2):
        if (word2 != "<s>"):
            max_arg = (self.final_bigrams[(word1,word2)]-self.disc_factor)
            #prob =  (max(max_arg,0)/self.unigram[word1])+ ((self.disc_factor()/self.unigram[word1])*self.s_value(word1)*self.smoothed_unigram.prob(word2))
            num = (self.disc_factor*self.s_value(word1)*self.prob_c(word2))
            prob =  (max(max_arg,0)+num)/self.unigram[word1]
            #print(max(max_arg,0),word1,word2)
            #print(self.unigram[word1])
            #print(self.disc_factor())
            #print(self.s_value(word1))
            #print(self.prob_c(word2))
            
            return prob
        else:
            return 0
        #enddef
        
    def prob_c(self,word2):
        den = len(self.bigrams)
        num_list = []
        for (w1,w2) in self.bigrams.keys():
            if (w2 == word2):
                num_list.append(w1)
        unique_num_list = set(num_list)
        num = len(unique_num_list)
        prob_c = num/den
        #print("num")
        #print(num,word2)
        #print("den")
        #print(den,word2)
        #print(prob_c)
        return prob_c
                            
    def s_value(self,word1):
        w2list = []
        for (w1,w2) in self.bigrams.keys():
            if (w1 == word1):
                w2list.append(w2)
        w2_uniquelist = set(w2list)
        s_value = len(w2_uniquelist)
        #print ("S value is")
        #print(s_value)
        return s_value
        
    # Generate a single random word according to the distribution
    def draw(self,word1= "<s>"):
        rand = random.random()
        for (w1,w2) in self.bigrams.keys():
            if (w1 == word1):
                rand -= self.prob(w1,w2)
                if rand <= 0.0:
                    return w2
	    #endif
	#endfor
    #enddef
#endclass
#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    #read your corpora
    trainCorpus = readFileToCorpus('pos_test.txt')
    trainCorpus = preprocess(trainCorpus)
    #posTestCorpus = readFileToCorpus('pos_test.txt')
    #negTestCorpus = readFileToCorpus('neg_test.txt')
    #vocab = set()
    #print("""Task 0: create a vocabulary(collection of word types) for the train corpus""")
    #posTestCorpus = preprocessTest(vocab, posTestCorpus)
    #negTestCorpus = preprocessTest(vocab, negTestCorpus)
    
    # Run sample unigram dist code
    # Sample test run for unigram model
    #unigram = UnigramModel(trainCorpus)
    #sbk = SmoothedBigramModelKN(trainCorpus)
    #sba = SmoothedBigramModelAD(trainCorpus)
    #bigram = BigramModel(trainCorpus)
    #sm = SmoothedUnigramModel(trainCorpus)
    # Task 1   (*** remember to generate 20 sentences for final output ***)
    #unigram.generateSentencesToFile(20, "unigram_output.txt")
    # Task 2
    #posTestCorpus = readFileToCorpus('pos_test.txt')
    #negTestCorpus = readFileToCorpus('neg_test.txt')
    #trainPerp = unigram.getCorpusPerplexity(trainCorpus)
    #posPerp = bigram.getCorpusPerplexity(posTestCorpus)
    #negPerp = bigram.getCorpusPerplexity(negTestCorpus)   
    #posPerp_s = sm.getCorpusPerplexity(posTestCorpus)
    #negPerp_s = sm.getCorpusPerplexity(negTestCorpus)   
    #print("Perplexity of positive training corpus:    "+ str(trainPerp))
    #print("Perplexity of positive review test corpus: "+ str(posPerp))
    #print("Perplexity of negative review test corpus: "+ str(negPerp))
    #print("Perplexity of positive review test corpus: "+ str(posPerp_s))
    #print("Perplexity of negative review test corpus: "+ str(negPerp_s))

    ## Fill in the functionality for SmoothedUnigramModel, BigramModel and SmoothedBigramModel, as well
    #smoothUnigram = SmoothedUnigramModel(trainCorpus)
    #smoothUnigram.generateSentencesToFile(20, "smooth_unigram_output.txt")  
    #bigram = BigramModel(trainCorpus)
    #bigram.generateSentencesToFile(20, "bigram_output.txt")  
    smoothBigramAD = SmoothedBigramModelAD(trainCorpus)
    p = smoothBigramAD.generateSentence() 
    print(p)
    #smoothBigramKN = SmoothedBigramModelKN(trainCorpus)
    #smoothBigramKN.generateSentencesToFile(20, "smooth_bigram_kn_output.txt")  
    

 