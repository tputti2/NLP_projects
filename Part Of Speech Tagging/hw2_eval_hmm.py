########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Evaluate the output of your bigram HMM POS tagger
##
import os.path
import sys
from operator import itemgetter
import hw2_hmm as hmm
import numpy as np

# A class for evaluating POS-tagged data
class Eval:
    ################################
    #input:                       #
    #    goldFile: string          #
    #    testFile: string          #
    #output: None                  #
    ################################
    def __init__(self, goldFile, testFile):
        print("Your task is to implement an evaluation program for POS tagging")
        #print(goldFile)
        check = hmm.HMM(5)
        #check.train('train.txt')
        self.check = check.readLabeledData(goldFile)
        #check.test('test.txt','output.txt')
        self.my_output = check.readLabeledData(testFile)
    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getTokenAccuracy(self):
        print("Return the percentage of correctly-labeled tokens")
        correct_token = 0
        incorrect_token = 0
        for i in range(len(self.check)):
            for j in range(len(self.check[i])):
                if self.check[i][j].word == self.my_output[i][j].word:
                    if self.check[i][j].tag == self.my_output[i][j].tag:
                        correct_token +=1
                    else:
                        incorrect_token +=1
                else:
                    print("the words do not match")
        accuracy = float(correct_token/(correct_token+incorrect_token))
        return accuracy

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getSentenceAccuracy(self):
        #print("Return the percentage of sentences where every word is correctly labeled")
        correct_sentence = 0
        incorrect_sentence = 0
        correct_token = np.zeros(len(self.check))
        for i in range(len(self.check)):
            for j in range(len(self.check[i])):
                if self.check[i][j].word == self.my_output[i][j].word:
                    if self.check[i][j].tag == self.my_output[i][j].tag:
                        correct_token[i] +=1
                else:
                    print("the words do not match")
            if correct_token[i] == len(self.check[i]):
                correct_sentence += 1
            else:
                incorrect_sentence += 1
        accuracy = float(correct_sentence/(correct_sentence+incorrect_sentence))
        return accuracy

    ################################
    #intput:                       #
    #    outFile: string           #
    #output: None                  #
    ################################
    def writeConfusionMatrix(self, outFile):
        #print("Write a confusion matrix to outFile; elements in the matrix can be frequencies (you don't need to normalize)")
        f=open(outFile, 'w+')
        
        confusion_matrix_first_row = []
        for i in range(len(self.check)):
            for j in range(len(self.check[i])):
                if self.check[i][j].tag not in confusion_matrix_first_row:
                    confusion_matrix_first_row.append(self.check[i][j].tag)

        bulk_matrix = np.zeros((len(confusion_matrix_first_row),len(confusion_matrix_first_row)),dtype=np.int)
        for i in range(len(self.check)):
            for j in range(len(self.check[i])):
                    a = confusion_matrix_first_row.index(self.check[i][j].tag)
                    b = confusion_matrix_first_row.index(self.my_output[i][j].tag)
                    bulk_matrix [a] [b] += 1
        #print(bulk_matrix.shape)
        confusion_matrix_first_row_array = np.array(confusion_matrix_first_row).reshape(-1,1)
        bulk_matrix = np.hstack((confusion_matrix_first_row_array,bulk_matrix))
        #print(bulk_matrix.shape)

        self.confusion_matrix = [[] for i in range(len(confusion_matrix_first_row)+1)]
        self.confusion_matrix [0] = confusion_matrix_first_row
        for i in range(len(confusion_matrix_first_row)):  
            for x in range(len(confusion_matrix_first_row)+1): 
                self.confusion_matrix [i+1].append(bulk_matrix[i][x])
          
        #print(self.confusion_matrix )
        for i in self.confusion_matrix :
            f.write("%s\n" %i)
                        
    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    def getPrecision(self, tagTi):
        #print("Return the tagger's precision when predicting tag t_i")
        self.writeConfusionMatrix("conf_matrix.txt")
        position = self.confusion_matrix[0].index(tagTi)
        TP = int(self.confusion_matrix[position+1][position+1])
        TFP = 0
        #print(position)
        for i in range(1,len(self.confusion_matrix)):
            #print(self.confusion_matrix[i][position+1])
            TFP += int(self.confusion_matrix[i][position+1])
        precision = float(TP/TFP)
        return precision

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    # Return the tagger's recall on gold tag t_j
    def getRecall(self, tagTj):
        #print("Return the tagger's recall for correctly predicting gold tag t_j")
        self.writeConfusionMatrix("conf_matrix.txt")
        position = self.confusion_matrix[0].index(tagTj)
        TP = int(self.confusion_matrix[position+1][position+1])
        TFN = 0
        for i in range(1,len(self.confusion_matrix)):
            TFN += int(self.confusion_matrix[position+1][i])
        recall = float(TP/TFN)
        return recall

if __name__ == "__main__":
    # Pass in the gold and test POS-tagged data as arguments
    if len(sys.argv) < 2:
        print("Call hw2_eval_hmm.py with two arguments: gold.txt and test.txt")
    else:
        gold = sys.argv[1]
        test = sys.argv[2]
        # You need to implement the evaluation class
        eval = Eval(gold, test)
        # Calculate accuracy (sentence and token level)
        print("Token accuracy: ", eval.getTokenAccuracy())
        print("Sentence accuracy: ", eval.getSentenceAccuracy())
        # Calculate recall and precision
        print("Recall on tag NNP: ", eval.getPrecision('NNP'))
        print("Precision for tag NNP: ", eval.getRecall('NNP'))
        # Write a confusion matrix
        eval.writeConfusionMatrix("conf_matrix.txt")

