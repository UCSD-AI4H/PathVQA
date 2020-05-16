from utils import *
from similarity import *

def evaluator(candidate, references, n, weights):

    if n == 0:
        return "0 (warning: n is 0)"
    elif len(split_sentence(candidate, 1)) == 0:
        return "0 (warning: length of candidate is 0)"
    elif len(split_sentence(candidate, n)) == 0:
        return "0 (warning: the n is too big)"
    else:
        return bleu(candidate, references, n, weights)

if __name__ == "__main__":
    candidate = "It is a guide to action which ensures that the military always obeys the commands of the party"
    references = ["It is a guide to action that ensures that the military will forever heed Party commands",
                  "It is the guiding principle which guarantees the military forces always being under the command of the Party",
                  "It is the practical guide for the army always to heed the directions of the party"]
    n = 4
    weights = [0.25, 0.25, 0.25, 0.25]
    
    if n == 0:
        print("warning: n is 0")
    elif len(split_sentence(candidate, 1)) == 0:
        print("warning: length of candidate is 0")
    else:
        print("belu is {}".format(bleu(candidate, references, n, weights)))