# Hidden Markov Model Part of Speech Taggers

#### Goal
To implement a Hidden Markov Model (HMM) model for Part Of Speech tagging. Specifically, you need to accomplish the following:
1. implement the Viterbi algorithm for a bigram HMM tagger
2. train this tagger on labeled training data (train.txt), and use it to tag unseen test data (test.txt).
3. write another program that compares the output of your tagger with the gold standard for the test data (gold.txt) to compute the overall token accuracy of your tagger. In addition, you need to compute the precision and recall for each tag and produce a confusion matrix.

To use pointwise mutual information to identify correlated pairs of words within the corpora.

#### Data
You will use three data files to train your tagger, produce Viterbi tags for unlabeled data, and evaluate your tagger's Viterbi output:
1. The file train.txt consists of POS-tagged text
2. The file test.txt consists of raw text. Each line is a single sentence, and     words are separated by spaces.
3. Once you finish your implementation, run your tagger over test.txt and         create a file out.txt that contains the Viterbi output.For evaluation purposes,     you need to compare the output of your tagger with gold.txt.





