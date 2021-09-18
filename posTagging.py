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

if __name__ == "__main__":
    nltk.download('brown')
    nltk.download('universal_tagset')
    corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
    mat_state, mat_obs, init_state = genMatrix(corpus)