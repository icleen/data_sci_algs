import numpy as np
import re
import scipy.stats as stats
import math
import time
import pickle

vocab = set()
docs = []

D = 472 # number of documents
K = 10 # number of topics

# open each file; convert everything to lowercase and strip non-letter symbols; split into words
for fileind in range( 1, D+1 ):
    foo = open( 'data/genconf/output%04d.txt' % fileind ).read()
    tmp = re.sub( '[^a-z ]+', ' ', foo.lower() ).split()
    docs.append( tmp )

    for w in tmp:
        vocab.add( w )

# vocab now has unique words
# give each word in the vocab a unique id
vhash = {}
vindhash = {}
for ind, i in enumerate(list(vocab)):
    vhash[i] = ind
    vindhash[ind] = i

# size of our vocabulary
V = len(vocab)

# reprocess each document and re-represent it as a list of word ids
docs_i = [[vhash[w] for w in d] for d in docs]

alphas = np.ones((K,1))[:,0]
gammas = np.ones((V,1))[:,0]

qs = [np.random.choice(range(K), size=(len(docs_i[d]))) for d in range(D)]
civk = np.zeros((D, V, K))
for d in range(D):
    for l in range(len(docs_i[d])):
        civk[d, docs_i[d][l], qs[d][l]] += 1

Li = np.sum(civk, axis=(1,2))

cvk = np.sum(civk, axis=0)
cik = np.sum(civk, axis=1)
ck = np.sum(cvk, axis=0)

probs = []

for iters in range(0,100):
    start_time = time.time()
    prob = 0
    for d in range(D):
        for l, word in enumerate(docs_i[d]):
            civk[d][word][qs[d][l]] -= 1
            cvk[word][qs[d][l]] -= 1
            cik[d][qs[d][l]] -= 1
            ck[qs[d][l]] -= 1

            prob += math.log(((cvk[word][qs[d][l]] + gammas[0]) / (ck[qs[d][l]] + V*gammas[0])) * ((cik[d][qs[d][l]] + alphas[0]) / (Li[d] + K*alphas[0])))
            pk = ((cvk[word] + gammas[0]) / (ck + V*gammas[0])) * ((cik[d] + alphas[0]) / (Li[d] + K*alphas[0]))
            pk /= np.sum(pk)
            nt = np.random.choice(range(K), size=(1), p=(pk))
            qs[d][l] = nt

            ck[qs[d][l]] += 1
            cik[d][qs[d][l]] += 1
            cvk[word][qs[d][l]] += 1
            civk[d][word][qs[d][l]] += 1

    print 'time: %f'%(time.time() - start_time)
    print("Iter %d, prob=%.2f" % (iters,prob))
    probs.append(prob)

with open('collapsed_probs.txt', 'w') as f:
    pickle.dump(probs, f)

with open('collapsed_qs.txt', 'w') as f:
    pickle.dump(qs, f)

with open('collapsed_civk.txt', 'w') as f:
    pickle.dump(civk, f)
