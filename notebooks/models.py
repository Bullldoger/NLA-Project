import numpy as np
import scipy as scp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from collections import Counter

class Word2Vec(object):
    
    def __init__(self, sentences):
        """
        sentences -- preprocessed sentences of reviews
        vocab -- vocabulary of a corpus words; {words: index}
        D -- word-context co-occurence matrix
        W -- matrix of words embeddings
        d -- dimension of words and reviews embeddings
        """

        self.sentences = sentences
        self.vocab = None
        self.D = None
        self.W = None
        self.d = 200
    
    ###### Create vocabulary from given sentences ######
    
    def create_vocabulary(self, r=200):
        self.vocab = dict()
        word_count = dict()
        idx = 0
        
        print('Creating vocabulary')
        for sentence in self.sentences:
            for word in sentence:
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1

        for word, count in word_count.items():
            if word_count[word] >= r:
                self.vocab[word] = idx
                idx += 1
                
        
    
    ###### Create word-context co-occurence matrix ######
    
    def create_corpus_matrix(self, L=2):
        print('Creating corpus matrix')
        # initialization
        words_counts = Counter()
        for sentence_index, sentence in enumerate(self.sentences):
            for word_index, word in enumerate(sentence):
                if word in self.vocab:
                    around_indexes = [i for i in range(max(word_index - L, 0), 
                                                       min(word_index + L + 1, len(sentence))) 
                                      if i != word_index]
                    for occur_word_index in around_indexes:
                            occur_word = sentence[occur_word_index]
                            if occur_word in self.vocab:
                                skipgram = (word, occur_word)
                                if skipgram in words_counts:
                                    words_counts[skipgram] += 1
                                else:
                                    words_counts[skipgram] = 1
        rows = list()
        cols = list()
        values = list()


        for (word_1, word_2), sharp in words_counts.items():                                            
            rows.append(self.vocab[word_1])
            cols.append(self.vocab[word_2])
            values.append(sharp)

        self.D = scp.sparse.csr_matrix((values, (rows, cols)))

    ###### Create matrix of words embeddings (as matrix factorization)######
        
    def compute_embeddings(self, k, alpha=.5):
        print('Computing of words embeddings')
        all_observations = self.D.sum()

        rows = []
        cols = []
        sppmi_values = []

        sum_over_words = np.array(self.D.sum(axis=0)).flatten()
        sum_over_contexts = np.array(self.D.sum(axis=1)).flatten()

        for word_index_1, word_index_2 in zip(self.D.nonzero()[0], 
                                                  self.D.nonzero()[1]):

            sg_count = self.D[word_index_1, word_index_2]

            pwc = sg_count
            pw = sum_over_contexts[word_index_1]
            pc = sum_over_words[word_index_2]

            spmi_value = np.log2(pwc * all_observations / (pw * pc * k))
            sppmi_value = max(spmi_value, 0)

            rows.append(word_index_1)
            cols.append(word_index_2)
            sppmi_values.append(sppmi_value)

        sppmi_mat = scp.sparse.csr_matrix((sppmi_values, (rows, cols)))
        U, S, V = scp.sparse.linalg.svds(sppmi_mat, self.d)
        self.W = U @ scp.diag(np.power(S, alpha))
        
    ###### Get vector embedding for a given word ######
    
    def get_word_embedding(self, word):
        if word in self.vocab:
            idx = self.vocab[word]
            return self.W[idx, :]
        else:
            print('This word is not in the vocabulary')
    