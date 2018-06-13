# Constant for NULL word at position zero in target sentence
NULL = "NULL"
import math
import numpy as np

# Your task is to finish implementing IBM Model 1 in this class
class IBMModel1:

    def __init__(self, trainingCorpusFile):
        # Initialize data structures for storing training data
        self.fCorpus = []                   # fCorpus is a list of foreign (e.g. Spanish) sentences

        self.tCorpus = []                   # tCorpus is a list of target (e.g. English) sentences

        self.trans = {}                     # trans[e_i][f_j] is initialized with a count of how often target word e_i and foreign word f_j appeared together.

        # Read the corpus
        self.initialize(trainingCorpusFile);

        # Initialize any additional data structures here (e.g. for probability model)

    # Reads a corpus of parallel sentences from a text file (you shouldn't need to modify this method)
    def initialize(self, fileName):
        f = open(fileName)
        i = 0
        j = 0;
        tTokenized = ();
        fTokenized = ();
        for s in f:
            if i == 0:
                tTokenized = s.split()
                # Add null word in position zero
                tTokenized.insert(0, NULL)
                self.tCorpus.append(tTokenized)
            elif i == 1:
                fTokenized = s.split()
                self.fCorpus.append(fTokenized)
                for tw in tTokenized:
                    if tw not in self.trans:
                        self.trans[tw] = {};
                    for fw in fTokenized:
                        if fw not in self.trans[tw]:
                             self.trans[tw][fw] = 1
                        else:
                            self.trans[tw][fw] =  self.trans[tw][fw] +1
            else:
                i = -1
                j += 1
            i +=1
        f.close()
        #print(self.tCorpus)
        #print(self.fCorpus)
        #print(self.trans)
        return

    # Uses the EM algorithm to learn the model's parameters
    def trainUsingEM(self, numIterations=10, writeModel=False, convergenceEpsilon = 0.01):#0.01):
        ###
        # Part 1: Train the model using the EM algorithm
        #
        # <you need to finish implementing this method's sub-methods>
        #
        ###

        # Compute translation length probabilities q(m|n)
        self.computeTranslationLengthProbabilities()         # <you need to implement computeTranslationlengthProbabilities()>
        # Set initial values for the translation probabilities p(f|e)
        self.initializeWordTranslationProbabilities()        # <you need to implement initializeTranslationProbabilities()>
        # Write the initial distributions to file
        if writeModel:
            self.printModel('initial_model.txt')                 # <you need to implement printModel(filename)>
        result = [0]
        for i in range(1,500):
            print ("Starting training iteration "+str(i))
            # Run E-step: calculate expected counts using current set of parameters
            self.computeExpectedCounts()                     # <you need to implement computeExpectedCounts()>
            # Run M-step: use the expected counts to re-estimate the parameters
            self.updateTranslationProbabilities()            # <you need to implement updateTranslationProbabilities()>
            # Write model distributions after iteration i to file
            if writeModel:
                self.printModel('model_iter='+str(i)+'.txt')     # <you need to implement printModel(filename)>
            final_log = 0
            for s in range(len(self.tCorpus)):
                n = len(self.tCorpus[s])
                m = len(self.fCorpus[s])
                b = 0
                for f in self.fCorpus[s]:
                    a = 0
                    for e in self.tCorpus[s]:
                        a += self.trans_prob[e][f]
                    b += math.log(a)

                final_log += math.log(self.length_prob[n][m]) - m*(math.log(n+1)) +b #Check if you should give n-1
            result.append(final_log/len(self.tCorpus))
            print(result[i]-result[i-1])
            if (abs(result[i] - result[i-1]) < convergenceEpsilon):
                break

    # Compute translation length probabilities q(m|n)
    def computeTranslationLengthProbabilities(self): # Change n to n-1
        # Implement this method
        self.length_prob = {}
        length = {}
        #den_length_prob = {}
        for s in range(len(self.fCorpus)):
            n = len(self.tCorpus[s])
            m = len(self.fCorpus[s])
            if n not in length:
                length[n] = {}

            if m not in length[n]:
                length[n][m] = 1

            length[n][m] += 1

        #print(length)

        for n, b in length.items():
            #print(n,b)
            self.length_prob[n] = {}
            for m, c in b.items():
                #print(m,c,b.values(),sum(b.values()))
                self.length_prob[n][m] = float(c) / float(sum(b.values()))
        #print(self.length_prob)
            #if (m,n) not in num_length_prob:
                #num_length_prob[m,n] = 1
            #else:
                #num_length_prob[m,n] += 1

            #if n not in den_length_prob:
                #den_length_prob[n] = 1
            #else:
                #den_length_prob[n] += 1

        #for (m,n) in num_length_prob:
            #self.length_prob[m,n] = float(num_length_prob[m,n])/float(den_length_prob[n])
            #if (self.length_prob[m,n] > 1) or (self.length_prob[m,n] <= 0):
                #print("You are calculating Length probabilities wrong")
        #print(self.length_prob)
        #pass


    # Set initial values for the translation probabilities p(f|e)
    def initializeWordTranslationProbabilities(self):
        # Implement this method
        self.trans_prob = {}
        for e, b in self.trans.items():
            self.trans_prob[e] = {}
            for f in b.keys():
                self.trans_prob[e][f] = float(1 / float(len(b.keys())))
        #print(self.trans_prob)
        #for e in self.trans.keys():
            #for f in self.trans[e].keys():
                #self.trans_prob[f,e] = float(1/float(len(self.trans[e])))
                #if e == 'yes':
                    #print((f,e),self.trans_prob[f,e])
        #print(self.trans_prob['No','go'])#pass

    # Run E-step: calculate expected counts using current set of parameters
    def computeExpectedCounts(self):
        #num = {}
        self.expected_count = {}
        for s in range(len(self.fCorpus)):
            for f in self.fCorpus[s]:
                den = 0
                for e in self.tCorpus[s]:
                    den+= self.trans_prob[e][f]
                for e in self.tCorpus[s]:
                    #self.expected_count[(s,f,e)] = float(num[(s,f,e)])/ float(den)
                    if e not in self.expected_count:
                        self.expected_count[e] = {}
                    if f not in self.expected_count[e]:
                        self.expected_count[e][f] = []
                    self.expected_count[e][f].append({'sentence': s, 'count': float(self.trans_prob[e][f])/ float(den)})
        #print(self.expected_count)
        #Implement this method
        #pass

    # Run M-step: use the expected counts to re-estimate the parameters
    def updateTranslationProbabilities(self):
        #norm_factor = {}
        #num = {}
        #for (s,j,i) in self.expected_count:
            #if self.tCorpus[s][i] in norm_factor:
                #norm_factor[self.tCorpus[s][i]] += self.expected_count[(s,j,i)]
            #else:
                #norm_factor[self.tCorpus[s][i]] = self.expected_count[(s,j,i)]

            #if (self.fCorpus[s][j], self.tCorpus[s][i]) in num:
                #num[self.fCorpus[s][j], self.tCorpus[s][i]] += self.expected_count[(s, j, i)]
            #else:
                #num[self.fCorpus[s][j], self.tCorpus[s][i]] = self.expected_count[(s, j, i)]

        #norm_factor = {}
        #num = {}
        #for (s,f,e) in self.expected_count:
            #if e in norm_factor:
                #norm_factor[e] += self.expected_count[(s, f, e)]
            #else:
                #norm_factor[e] = self.expected_count[(s, f, e)]

            #if (f,e) in num:
                #num[f, e] += self.expected_count[(s, f, e)]
            #else:
                #num[f, e] = self.expected_count[(s, f, e)]

        #for e in self.trans.keys():
            #for f in self.trans[e].keys():
                #self.trans_prob[f, e] = float(num[f, e])/ float(norm_factor[e])
            # Implement this method

        for e in self.trans.keys():
            norm_factor = {}
            for f in self.trans[e].keys():
                norm_factor[f] = 0
                for s in self.expected_count[e][f]:
                    norm_factor[f] += s['count']

            den = sum(norm_factor.values())

            for f in self.trans[e].keys():
                self.trans_prob[e][f] = norm_factor[f] / den
        #pass

    # Returns the best alignment between fSen and tSen, according to your model
    def align(self, fSen, tSen):
        ###
        # Part 2: Find and return the best alignment
        # <you need to finish implementing this method>
        # Remove the following code (a placeholder return that aligns each foreign word with the null word in position zero of the target sentence)
        ###
        m = len(fSen)

        #Assign everything to NULL first
        a = [0]*m
        #if tSen[0] not in 'NULL' :
            #tSen.insert(0, NULL)
        n = len(tSen)
        #print(tSen)
        #Keep changing one by one
        #import itertools
        #a = list(itertools.product((x for x in range(n+1)),repeat = m))
        #print(m,n,len(a))

        for j in range(m):
           best_align = -1
           for i in range(n):
                if  best_align < self.trans_prob[tSen[i]][fSen[j]]:
                    best_align =  self.trans_prob[tSen[i]][fSen[j]]
                    a[j] = int(i)
        #print(a)
        #print(tSen)
        #print(fSen)

        return a   # Your code above should return the correct alignment instead

    # Return q(tLength | fLength), the probability of producing an English sentence of length tLength given a non-English sentence of length fLength
    # (Can either return log probability or regular probability)
    def getTranslationLengthProbability(self, fLength, tLength):
        result = self.length_prob[tLength][fLength]
        # Implement this method
        return result

    # Return p(f_j | e_i), the probability that English word e_i generates non-English word f_j
    # (Can either return log probability or regular probability) #Change the position of this to original
    def getWordTranslationProbability(self, f_j, e_i):
        result = self.trans_prob[e_i][f_j]
        # Implement this method
        return result

    # Write this mo,k[del's probability distributions to file
    def printModel(self, filename):
        # Write q(m|n) for all m,n to this file
        lengthFile = open(filename+'_lengthprobs.txt', 'w')
        for n,b in self.length_prob.items():
            for m, c in b.items():
                lengthFile.write("Values of m,n and q(m|n) are %d, %d and %.7f respectively\n" %(m,n, self.length_prob[n][m]))


        # Write p(f_j | e_i) for all f_j, e_i to this file
        translateProbFile = open(filename + '_translationprobs.txt', 'w')
        for e,b in self.trans_prob.items():
            for f,c in b.items():
                translateProbFile.write("Values of f,e and p(f|e) are %s, %s and %.7f respectively\n" % (f,e,self.trans_prob[e][f]))

        lengthFile.close();
        translateProbFile.close()
        # Implement this method (make your output legible and informative)



# utility method to pretty-print an alignment
# You don't have to modify this function unless you don't think it's that pretty...
def prettyAlignment(fSen, tSen, alignment):
    pretty = ''
    for j in range(len(fSen)):
        pretty += str(j)+'  '+fSen[j].ljust(20)+'==>    '+tSen[alignment[j]]+'\n';
    return pretty

if __name__ == "__main__":
    # Initialize model
    model = IBMModel1('eng-spa.txt')
    # Train model
    model.trainUsingEM(10,writeModel=True);
    model.printModel('after_training')
    # Use model to get an alignment
    fSen = 'No pierdas el tiempo por el camino .'.split()
    tSen = 'Don\' t dawdle on the way'.split()
    alignment = model.align(fSen, tSen);
    print (prettyAlignment(fSen, tSen, alignment))
