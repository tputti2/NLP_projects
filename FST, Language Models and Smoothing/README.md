# Language Models and Smoothing

#### Goal
1. To train some n-gram language models - unigram, bigram, smoothed unigram (using Laplace smoothing), smoothed bigram model (using Absolute Discounting and Kneser-Ney smoothing algorithms) on a corpus of movie reviews and to test them on two smaller corpora: a collection of positive reviews, and one of negative reviews.
2. To implement a finite-state transducer (or FST) which transduces the infinitive form of verbs to their correct -ing form. 
    1. Dropping the final -e:
    ride ==> riding   
    2. Double the final consonant:  stop ==> stopping
    3. Change final -ie to -y :
    die ==> dying
#### Data
For Goal 1: 
1. train.txt: contains the training corpus of movie reviews (30k sentences)
2. pos_test.txt: contains the test corpus of positive reviews (1k sentences)
3. neg_test.txt: contains the test corpus of negative reviews (1k sentences)

For Goal 2:

1. 360verbs.txt: sample input file containing a list of 360 infinitive verbs, one verb per line.
2. 360verbsCorrect.txt: sample output file containing the correct translation (infinitive to -ing form) for each of the 360 verbs in order, one translation per line.
