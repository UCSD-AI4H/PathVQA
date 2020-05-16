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
    n = 4
    weights = [0.25, 0.25, 0.25, 0.25]
    
    if n == 0:
        print("warning: n is 0")
    elif len(split_sentence(candidate, 1)) == 0:
        print("warning: length of candidate is 0")
    else:
        print("belu is {}".format(bleu(candidate, references, n, weights)))