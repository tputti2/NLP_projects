# Context Free grammar and Sentence Parsing

#### Goal

1. To write a context-free grammar that can parse a set of short sentences. Your
task is to define the rules for your CFG in mygrammar.cfg. i.e. you will need to add rules such as:

    S -> NP VP 

    NP -> DT N

    N -> NN | NNS | ...
    etc.

2. Your second task is to implement the CKY algorithm for parsing using a probabilistic context-free grammar (PCFG). Given a PCFG, the algorithm uses dynamic programming to return the most-likely parse for an input sentence.

#### Data
We have given you a text file (toygrammar.pcfg) containing a toy grammar that generates sentences similar to the running examples seen in class. Each binary rule in the grammar is stored on a separate line, in the following format:

prob P -> LC RC

where prob is the rule's probability, P is the left-hand side of the rule (a parent nonterminal), and LC and
RC are the left and right children, respectively.
For unary rules, we only have a single child C and the line has format:

prob P -> C
