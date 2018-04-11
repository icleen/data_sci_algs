
# coding: utf-8

# In[1]:


import sys
import numpy as np
import math


# In[2]:


def get_input_numbers():
    v = raw_input('Please enter a set of numbers: ')
    if ", " in v:
        arr = v.split(', ')
    elif "," in v:
        arr = v.split(',')
    else:
        arr = v.split(' ')
    ints = [int(a) for a in arr]
    return np.array(ints)


# In[3]:


def in_hyp(hyp, val):
    switcher = {
        "even": lambda x: x % 2 == 0, 
        "odd": lambda x: x % 2 == 1, 
        "squares": lambda x: x == 1 or x == 4 or x == 9 or x == 16 or x == 25 or x == 36 or x == 49 or x == 64 or x == 81 or x == 100, 
        
        "mult of 3": lambda x: x % 3 == 0, 
        "mult of 4": lambda x: x % 4 == 0, 
        "mult of 5": lambda x: x % 5 == 0,
        "mult of 6": lambda x: x % 6 == 0, 
        "mult of 7": lambda x: x % 7 == 0, 
        "mult of 8": lambda x: x % 8 == 0,  
        "mult of 9": lambda x: x % 9 == 0, 
        "mult of 10": lambda x: x % 10 == 0, 
            
        "ends in 1": lambda x: x % 10 == 1, 
        "ends in 2": lambda x: x % 10 == 2, 
        "ends in 3": lambda x: x % 10 == 3, 
        "ends in 4": lambda x: x % 10 == 4, 
        "ends in 5": lambda x: x % 10 == 5,
        "ends in 6": lambda x: x % 10 == 6, 
        "ends in 7": lambda x: x % 10 == 7, 
        "ends in 8": lambda x: x % 10 == 8, 
        "ends in 9": lambda x: x % 10 == 9, 
        
        "powers of 2": lambda x: x == 2 or x == 4 or x == 8 or x == 16 or x == 32 or x == 64,
        "powers of 3": lambda x: x == 3 or x == 9 or x == 27 or x == 81, 
        "powers of 4": lambda x: x == 4 or x == 16 or x == 64, 
        "powers of 5": lambda x: x == 5 or x == 25, 
        "powers of 6": lambda x: x == 6 or x == 36, 
        "powers of 7": lambda x: x == 7 or x == 49,
        "powers of 8": lambda x: x == 8 or x == 64, 
        "powers of 9": lambda x: x == 9 or x == 81, 
        "powers of 10": lambda x: x == 10 or x == 100, 
        
        "all": lambda x: True, 
        "powers of 2 + {37}": lambda x: x == 2 or x == 4 or x == 8 or x == 16 or x == 32 or x == 64 or x == 37, 
        "powers of 2 - {32}": lambda x: x == 2 or x == 4 or x == 8 or x == 16 or x == 64,
    }
    
    func = switcher.get(hyp, lambda x: False)
    return func(val)

print in_hyp("powers of 3", 9)


# In[4]:


concepts = ["even", "odd", "squares", "mult of 3", "mult of 4", "mult of 5", 
            "mult of 6", "mult of 7", "mult of 8",  "mult of 9", "mult of 10", 
            "ends in 1", "ends in 2", "ends in 3", "ends in 4", "ends in 5",
            "ends in 6", "ends in 7", "ends in 8", "ends in 9", "powers of 2",
            "powers of 3", "powers of 4", "powers of 5", "powers of 6", "powers of 7",
            "powers of 8", "powers of 9", "powers of 10", "all", 
            "powers of 2 + {37}", "powers of 2 - {32}"]
print concepts


# In[5]:


data_over_hyp = np.zeros((len(concepts), 100))
for i, hyp in enumerate(concepts):
    for val in xrange(100):
        if in_hyp(hyp, val + 1):
            data_over_hyp[i,val] = 1


# In[6]:


prior = np.ones(( len(concepts), 1 ))
prior[0] = 5
prior[1] = 5
prior[2] = 5
prior[21] = 5
prior = prior / np.sum(prior)

divisor = np.array([1 / np.sum(data_over_hyp[i]) for i in xrange(len(concepts))])
print divisor


# In[7]:


values = get_input_numbers()


# In[8]:


# Get the likelihood for the input values from the data_over_hyp matrix
temp = np.array([data_over_hyp[:,val - 1] for val in values])

sum = np.zeros((len(concepts)))
for row in temp:
    sum += row

max = np.max(sum)
sum = map(lambda x: 0 if x != max else x, sum)

# print sum

sum = [math.pow(divisor[i], sum[i]) for i in range(len(sum))]
likelihood = np.array(map(lambda x: 0 if x == 1.0 else x, sum))

print likelihood


# In[9]:


# Get the posterior by multiplying the prior and the likelihood
posterior = np.multiply(likelihood, prior.reshape(len(concepts)))
# print posterior

# Normalize the posterior
posterior = posterior / np.sum(posterior)
print posterior


# In[15]:


result = np.dot(posterior, data_over_hyp)
print result.shape
print result

