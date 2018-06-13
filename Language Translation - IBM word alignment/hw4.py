from datetime import datetime

# Constant for NULL word at position zero in target sentence
NULL = "NULL"

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
        return

    # Uses the EM algorithm to learn the model's parameters
    def trainUsingEM(self, numIterations=10, writeModel=False, convergenceEpsilon=0.01):
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
        for i in range(numIterations):
            print ("Starting training iteration "+str(i))
            # Run E-step: calculate expected counts using current set of parameters
            self.computeExpectedCounts()                     # <you need to implement computeExpectedCounts()>
            # Run M-step: use the expected counts to re-estimate the parameters
            self.updateTranslationProbabilities()            # <you need to implement updateTranslationProbabilities()>
            # Write model distributions after iteration i to file
            if writeModel:
                self.printModel('model_iter='+str(i)+'.txt')     # <you need to implement printModel(filename)>

    # Compute translation length probabilities q(m|n)
    def computeTranslationLengthProbabilities(self):
        transLengthCounts = {}

        for i in range(len(self.fCorpus)):
            m = len(self.fCorpus[i])
            n = len(self.tCorpus[i])

            if n not in transLengthCounts:
                transLengthCounts[n] = {}

            if m not in transLengthCounts[n]:
                transLengthCounts[n][m] = 0

            transLengthCounts[n][m] += 1

        self.transLengthProbs = {}
        for m, counts in transLengthCounts.items():
            self.transLengthProbs[m] = {}
            for n, count in counts.items():
                self.transLengthProbs[m][n] = count / sum(counts.values())

    # Set initial values for the translation probabilities p(f|e)
    def initializeWordTranslationProbabilities(self):
        self.wordTransProbs = {}

        for tWord, counts in self.trans.items():
            self.wordTransProbs[tWord] = {}
            for fWord in counts.keys():
                self.wordTransProbs[tWord][fWord] = float(1 / float(len(counts.keys())))
        print(self.wordTransProbs['go']['No'])

    # Run E-step: calculate expected counts using current set of parameters
    def computeExpectedCounts(self):
        print('E-step...')
        print(datetime.now())
        #self.expectedCounts = []
        self.expectedCounts = {}
        for s in range(len(self.fCorpus)):
            for fj in self.fCorpus[s]:
                e_sum = 0
                for ei in self.tCorpus[s]:
                    e_sum += self.wordTransProbs[ei][fj]
                #print(e_sum)
                for ei in self.tCorpus[s]:
                    if ei not in self.expectedCounts:
                        self.expectedCounts[ei] = {}
                    if fj not in self.expectedCounts[ei]:
                        self.expectedCounts[ei][fj] = []
                    self.expectedCounts[ei][fj].append({'sentence' : s, 'count' : self.wordTransProbs[ei][fj] / e_sum})


        #input(datetime.now())
            #expectedCounts = {}

            # for fj in self.fCorpus[s]:
            #     e_sum = 0
            #     for ei in self.tCorpus[s]:
            #         e_sum += self.wordTransProbs[ei][fj]
            #     for ei in self.tCorpus[s]:
            #         if ei not in expectedCounts:
            #             expectedCounts[ei] = {}
            #         expectedCounts[ei][fj] = self.wordTransProbs[ei][fj] / e_sum

            # self.expectedCounts.append(expectedCounts)


            #DEL
            #     expectedCounts[ei] = {}
            #     e_sum = sum(self.wordTransProbs[ei].values())
            #     for fj in self.fCorpus[s]:
            #         expectedCounts[ei][fj] = self.wordTransProbs[ei][fj] / e_sum
            # self.expectedCounts.append(expectedCounts)

    # Run M-step: use the expected counts to re-estimate the parameters
    def updateTranslationProbabilities(self):
        print('M-step...')
        print(datetime.now())
        #input(len(self.trans.keys()))
        for ex in self.trans.keys():
            #compute normalization factor
            f_sum = {}
            for fy in self.trans[ex].keys():
                f_sum[fy] = 0
                for sentence in self.expectedCounts[ex][fy]:
                    f_sum[fy] += sentence['count']

            z = sum(f_sum.values())

            for fy in self.trans[ex].keys():
                self.wordTransProbs[ex][fy] = f_sum[fy] / z
        print(datetime.now())


        # for ex in self.trans.keys():
        #     #compute normalization factor
        #     f_sum = {}
        #     for fy in self.trans[ex].keys():
        #         f_sum[fy] = 0
        #         for sentence in self.expectedCounts:
        #             if ex in sentence and fy in sentence[ex]:
        #                 f_sum[fy] += sentence[ex][fy]

        #     z = sum(f_sum.values())

        #     for fy in self.trans[ex].keys():
        #         self.wordTransProbs[ex][fy] = f_sum[fy] / z
        # print(datetime.now())


    # Returns the best alignment between fSen and tSen, according to your model
    def align(self, fSen, tSen):
        ###
        # Part 2: Find and return the best alignment
        # <you need to finish implementing this method>
        # Remove the following code (a placeholder return that aligns each foreign word with the null word in position zero of the target sentence)
        ###
        #dummyAlignment = [0]*len(fSen)
        #input(dummyAlignment)
        #return dummyAlignment   # Your code above should return the correct alignment instead

        alignment = []
        for fy in fSen:
            e_max_prob = -1
            e_max_index = 0
            for ei in range(len(tSen)):
                if self.wordTransProbs[tSen[ei]][fy] > e_max_prob:
                    e_max_index = ei
                    e_max_prob = self.wordTransProbs[tSen[ei]][fy]
            #for i in range(tSen):
            #alignment
            alignment.append(e_max_index)

        return alignment

    # Return q(tLength | fLength), the probability of producing an English sentence of length tLength given a non-English sentence of length fLength
    # (Can either return log probability or regular probability)
    def getTranslationLengthProbability(self, fLength, tLength):
        return self.transLengthProbs[tLength][fLength]

    # Return p(f_j | e_i), the probability that English word e_i generates non-English word f_j
    # (Can either return log probability or regular probability)
    def getWordTranslationProbability(self, f_j, e_i):
        # Implement this method
        return self.wordTransProbs[e_i][f_j]

    # Write this model's probability distributions to file
    def printModel(self, filename):
        lengthFile = open(filename+'_lengthprobs.txt', 'w')         # Write q(m|n) for all m,n to this file
        translateProbFile = open(filename+'_translationprobs.txt', 'w') # Write p(f_j | e_i) for all f_j, e_i to this file
        
        for m, probs in self.transLengthProbs.items():
            for n, prob in probs.items():
                lengthFile.write('q({}|{}) = {}\n'.format(m, n, prob))

        for ei, probs in self.wordTransProbs.items():
            for fj, prob in probs.items():
                translateProbFile.write('p({}|{}) = {}\n'.format(fj, ei, prob))

        lengthFile.close();
        translateProbFile.close()

# utility method to pretty-print an alignment
# You don't have to modify this function unless you don't think it's that pretty...
def prettyAlignment(fSen, tSen, alignment):
    pretty = ''
    for j in range(len(fSen)):
        pretty += str(j)+'  '+fSen[j].ljust(20)+'==>    '+tSen[alignment[j]]+'\n';
    return pretty

if __name__ == "__main__":
    # Initialize model
    model = IBMModel1('eng-spa_small.txt')
    # Train model
    model.trainUsingEM(10);
    model.printModel('after_training')
    # Use model to get an alignment
    fSen = 'No pierdas el tiempo por el camino .'.split()
    tSen = 'Don\' t dawdle on the way'.split()
    alignment = model.align(fSen, tSen);
    print (prettyAlignment(fSen, tSen, alignment))
