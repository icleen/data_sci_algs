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

cvk = np.sum(civk, axis=0)
cik = np.sum(civk, axis=1)
ck = np.sum(cvk, axis=0)

probs = []
pis2 = np.zeros((D, K))
max_iters = 100

for iters in range(0,max_iters):
    start_time = time.time()
    prob = 0

    # resample per-word topic assignments bs (K, V)
    bs = np.zeros((K, V))
    tmp = gammas + cvk.T
    for k in range(K):
        bs[k] = np.random.dirichlet(tmp[k])
        prob += stats.dirichlet.logpdf(bs[k], tmp[k])

    # resample topics (D, K) and per-document topic mixtures pis (D, K)
    for d in range(D):
        # resample topic mixture for document d
        pis =  np.random.dirichlet(alphas + cik[d])
        prob += stats.dirichlet.logpdf(pis, alphas + cik[d])
        if iters == max_iters-1:
            pis2[d] += pis
        for l, word in enumerate(docs_i[d]):
            theta = pis * bs[:, word]
            prob += math.log(theta[qs[d][l]])
            theta /= np.sum(theta)

            civk[d][word][qs[d][l]] -= 1
            cvk[word][qs[d][l]] -= 1
            cik[d][qs[d][l]] -= 1
            ck[qs[d][l]] -= 1
            # resample topic of word l in document d
            qs[d][l] = np.random.choice(range(K), size=(1), p=(theta))
            ck[qs[d][l]] += 1
            cik[d][qs[d][l]] += 1
            cvk[word][qs[d][l]] += 1
            civk[d][word][qs[d][l]] += 1


    print 'time: %f'%(time.time() - start_time)
    print("Iter %d, prob=%.2f" % (iters,prob))
    probs.append(prob)

with open('simple_probs', 'w') as f:
    pickle.dump(probs, f)
print 'simple_probs'

with open('simple_qs', 'w') as f:
    pickle.dump(qs, f)
print 'simple_qs'

with open('simple_bs', 'w') as f:
    pickle.dump(bs, f)
print 'simple_bs'

with open('simple_pis', 'w') as f:
    pickle.dump(pis2, f)
print 'simple_pis'

with open('simple_civk', 'w') as f:
    pickle.dump(civk, f)
print 'simple_civk'
