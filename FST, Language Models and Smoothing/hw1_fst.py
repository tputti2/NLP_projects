from fst import *

# here are some predefined character sets that might come in handy.
# you can define your own
AZ = set("abcdefghijklmnopqrstuvwxyz")
VOWS = set("aeiou")
CONS = set("bcdfghjklmnprstvwxyz")
CONSu = set("bcdfghjklmnprstvwxyzu")
E = set(["E","e"])
I = set(["I","i"])
U = set(["U","u"])
NPTR = set(["N","n","P","p","T","t","R","r"])
PT = set(["p","t"])

# Implement your solution here
def buildFST():
    print("Your task is to implement a better FST in the buildFST() function, using the methods described here")
    print("You may define additional methods in this module (hw1_fst.py) as desired")

    #f = FST("q0") # q0 is the initial (non-accepting) state
    #f.addState("q1") # a non-accepting state
    #f.addState("q_ing") # a non-accepting state
    #f.addState("q_EOW", True) # an accepting state (you shouldn't need any additional accepting states)

    #f.addState("q_cons")
    #f.addState("q_vow")
    f.addState("q_nptr")
    f.addState("q_e")
    f.addState("q_rn")
    #f.addState("q_u")
    f.addState("q_i")
    
    #f.addSetTransition("q0", CONS, "q_cons")
    #f.addSetTransition("q0", VOWS, "q_vow")
    #f.addSetTransition("q_vow", AZ, "q_cons")
    #f.addSetTransition("q_cons", CONS, "q_cons")

    #f.addEpsilonTransition("q_cons", "q_ing")
    #f.addTransition("q_cons", "e", "", "q_ing")

    #f.addState("q_p")
    #f.addSetTransition("q_cons", VOWS-set("i"), "q_p")
    #f.addTransition("q_p", "p", "pp" , "q_ing")

    #f.addState("q_t")
    #f.addSetTransition("q_cons", VOWS-set("i"), "q_t")
    #f.addTransition("q_t", "t", "tt" , "q_ing")

    #f.addTransition("q_cons", "i", "", "q_i")
    #f.addTransition("q_i", "e", "y",  "q_ing")

    #f.addState("x")
    #f.addSetTransition("q_cons", "u", "q_u")
    #f.addTransition("q_u", "e", "",  "q_ing")

    #f.addTransition("q_u", "n", "nn",  "q_ing")
    #f.addTransition("q_u", "r", "rr",  "q_ing")

    #f.addSetTransition("q_u", AZ-set("ent"), "q_cons")
    #f.addSetTransition("q_u", set("ent"), "x")
    #f.addSetTransition("x", AZ-E, "q_cons")
    #f.addSetTransition("x", E, "x")
    #f.addTransition("x", "e", "", "q_ing")

    f.addState("q_nr")
    #f.addSetTransition("q_cons", set("i"), "q_nptr")
    #f.addTransition("q_nptr", "n", "nn", "q_ing")
    #f.addTransition("q_nptr", "r", "rr", "q_ing")
    #f.addTransition("q_nptr", "t", "tt", "q_ing")
    #f.addTransition("q_nptr", "p", "pp", "q_ing")
    #f.addSetTransition("q_nptr", AZ-set("nptre"), "q_cons")
    #f.addSetTransition("q_nptr", set("nptr"), "x")

    #f.addState("q_n")
    #f.addSetTransition("q_cons", set("e"), "q_n")
    #f.addSetTransition("q_n", set("nr"), "q_ing")
    #f.addSetTransition("q_n", AZ-set("nrt"), "q_cons")
    #f.addSetTransition("q_n", set("nrt"), "x")

    #f.addState("q_ao")
    #f.addSetTransition("q_cons", set("ao"), "q_ao")
    #f.addTransition("q_ao", "n", "nn", "q_ing")
    #f.addTransition("q_ao", "r", "rr", "q_ing")
    #f.addSetTransition("q_ao", AZ-set("nptr"), "q_cons")
    #f.addSetTransition("q_ao", set("nptr"), "x")

    #add -ing
    #f.addTransition("q_ing", "", "ing", "q_EOW")
    
    #return f



    

if __name__ == "__main__":
    # Pass in the input file as an argument
    if len(sys.argv) < 2:
        print("This script must be given the name of a file containing verbs as an argument")
        quit()
    else:
        file = sys.argv[1]
    #endif

    # Construct an FST for translating verb forms 
    # (Currently constructs a rudimentary, buggy FST; your task is to implement a better one.
    f = buildFST()
    # Print out the FST translations of the input file
    f.parseInputFile('360verbs.txt')
