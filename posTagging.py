# print ("hello assignment 4!")

from typing import List
import nltk
import string
import numpy as np
from numpy.lib.index_tricks import index_exp


def genMatrix(_corpus):
    # count the unque words in the corpus, and make these words a list of mapping of idx and word
    wordMap, tagMap = getWordTagCount(_corpus)
    matState = np.zeros((len(tagMap), len(tagMap)))
    obsState = np.zeros((len(tagMap), len(wordMap)))
    initState = np.zeros(len(tagMap))

    # print (matState)
    # print ()
    # print (obsState)

    # separate corpus in sentence unit as a list, read one by one (for loop)
    for sentence in _corpus:
        # print (str(sentence) + "\n")

        # generate the State Matrix
        # read the element in a tuple unit, read one by one
        for tWordTag_first, tWordTag_second in zip(sentence[:-1], sentence[1:]):
            # print(tWordTag_first[1], tWordTag_second[1])
            # print("state_row_index: " + str(tagMap.index(tWordTag_first[1])))
            # print("state_col_index: " + str(tagMap.index(tWordTag_second[1])))

            # the tag, second element of tuple, to build the state matrix
            matState[
                tagMap.index(tWordTag_first[1]), # state_row_index
                tagMap.index(tWordTag_second[1]) # state_col_index
            ] += 1

            # print ("matState: \n" + str(matState) )
            pass

        # generate the Obsveration Matrix        
        for tWordtag in sentence:
            # print (tWordtag[0], tWordtag[1])
            # print("state_row_index: " + str(tagMap.index(tWordtag[1])))
            # print("state_col_index: " + str(wordMap.index(tWordtag[0])))

            # obsState
            # the word, first element of tuple, to build the observation matrix 
            obsState[
                tagMap.index(tWordtag[1]), # obs_row_index
                wordMap.index(tWordtag[0]), # obs_col_index
            ] += 1

            # print ("obsState: \n" + str(obsState) )
            pass
            
        # build the initial state distribution
        # Take the first element of the tuple of the 10000 sentences 
        print(sentence[0])
        # if (sentence[0])

        # # initState[0] = np.sum(obsState[0, :])/np.sum(obsState[:,:])

        pass

    # normalize matrix 
    # print ("matState: \n" + str(matState))
    # print ("mat denominator: \n" + str(np.sum(matState, axis=1, keepdims=True)))
    matState = matState/np.sum(matState, axis=1, keepdims=True)
    print ("normalized matState: \n" + str(matState))
    
    # np.set_printoptions(threshold=np.inf)
    # print ("obsState: \n" + str(obsState))
    # print ("obsState size: " + str(obsState.shape))
    # print ("obs denominator: \n" + str(np.sum(obsState, axis=1, keepdims=True)))
    # print ("obs denominator size: " + str(np.sum(obsState, axis=1, keepdims=True).shape))
    obsState = obsState/np.sum(obsState, axis=1, keepdims=True)
    print ("normalized obsState: \n" + str(obsState))

    return 0

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

    print ("List of unique word: \n" + str(word_map) + "\n")
    print ("Number of unique words: " + str(len(word_map)) + "\n")
    print ("List of unique tags: \n" + str(tag_map) + "\n")
    print ("Number of unique tags: " + str(len(tag_map)) + "\n")

    return word_map, tag_map

def genObsM():

    return

if __name__ == "__main__":
    nltk.download('brown')
    nltk.download('universal_tagset')
    # corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10]
    corpus = [[('The', 'DET'), ('Fulton', 'NOUN'), ('County', 'NOUN'), ('Grand', 'ADJ'), ('Jury', 'NOUN'), 
                ('The', 'NOUN'), ('Fulton', 'DET'), ('County', 'ADJ'), ('Grand', 'NOUN'), ('Jury', 'DET'), 
                ('The', 'NOUN'), ('Fulton', 'ADJ'), ('County', 'ADJ'), ('Grand', 'DET'), ('Jury', 'DET')]]
    # print (corpus)
    # print (type(corpus))
    # print (len(corpus), len(corpus[0]))
    # print (string.punctuation)

    genMatrix(corpus)
