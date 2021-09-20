from typing import List
import nltk
import string
import numpy as np
from numpy.lib.index_tricks import index_exp


def genMatrix(_corpus):
    # count the unque words in the corpus, and make these words a list of mapping of idx and word
    wordMap, tagMap = getWordTagCount(_corpus)
    matState = np.zeros((len(tagMap), len(tagMap)))
    matObs = np.zeros((len(tagMap), len(wordMap)))
    initState = np.zeros(len(tagMap))

    # separate corpus in sentence unit as a list, read one by one (for loop)
    for sentence in _corpus:
        # generate the State Matrix
        # read the element in a tuple unit, read one by one
        for tWordTag_first, tWordTag_second in zip(sentence[:-1], sentence[1:]):
            # the tag, second element of tuple, to build the state matrix
            matState[
                tagMap.index(tWordTag_first[1]), # state_row_index
                tagMap.index(tWordTag_second[1]) # state_col_index
            ] += 1
            pass
        # generate the Obsveration Matrix        
        for tWordtag in sentence:
            # matObs
            # the word, first element of tuple, to build the observation matrix 
            matObs[
                tagMap.index(tWordtag[1]), # obs_row_index
                wordMap.index(tWordtag[0]), # obs_col_index
            ] += 1
            pass  
        # build the initial state distribution
        # Take the first element of the tuple of the 10000 sentences 
        initState[tagMap.index(sentence[0][1])] += 1 
        pass
    # implementing smoothing for the 2 matrix and 1 list so that we won't enounter log0 issue
    matState += 1
    matObs += 1
    initState += 1

    # normalize matrix 
    matState = matState/np.sum(matState, axis=1, keepdims=True)
    matObs = matObs/np.sum(matObs, axis=1, keepdims=True)
    initState = initState/np.sum(initState, axis=0, keepdims=True)

    return matState, matObs, initState

def getWordTagCount(_corpus):
    word_map = []
    tag_map = []

    for sentence in _corpus:
        for tWordTag in sentence:
            word_map.append(tWordTag[0])
            tag_map.append(tWordTag[1])
            pass
        pass
    word_map = sorted(set(word_map))
    tag_map = sorted(set(tag_map))

    return word_map, tag_map

def viterbi(obs, pi, A, B):
    """
    Finds the viterbi solution for a hidden markov model 
    Returns -> list of ints
    Optimal state sequence corresponding to the viterbi solution
    Parameters: 
    obs -> List of ints
    List of observations
    pi -> numpy.array of dim (number of states)
    Initial probabilities of states
    A -> numpy.array (number of states, number of states)
    Transition Probability  Matrix
    B -> numpy.array(number of states, number of possible observations)
    Emmission Probability Marix
    """
    pi = np.log(pi)
    A = np.log(A)
    B = np.log(B)

    v = np.zeros([A.shape[0], len(obs)])
    r = np.zeros([A.shape[0], len(obs)]) - 1
    v = pi + B[:, obs[0]]
    r[:,0] = np.arange(A.shape[0])
    
    for t in range(1, len(obs)):
        vab = np.expand_dims(v, 1) + A + B[:,obs[t]:(obs[t]+1)]
        v = np.amax(vab, 0)
        r[:,t] = np.argmax(vab, 0)
        pass

    z = [0]*len(obs)
    z[-1] = v.argmax(0)
    for t in range(len(obs)-1, 0, -1):
        z[t-1] = int(r[z[t], t])
        pass
    return list(z)
    



if __name__ == "__main__":
    nltk.download('brown')
    nltk.download('universal_tagset')
    corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
    mat_state, mat_obs, init_state = genMatrix(corpus)
