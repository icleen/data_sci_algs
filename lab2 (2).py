
# coding: utf-8

# In[84]:


import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn


# In[85]:


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


# In[86]:


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


# In[87]:


concepts = np.array(["even", "odd", "squares", "mult of 3", "mult of 4", "mult of 5", 
            "mult of 6", "mult of 7", "mult of 8",  "mult of 9", "mult of 10", 
            "ends in 1", "ends in 2", "ends in 3", "ends in 4", "ends in 5",
            "ends in 6", "ends in 7", "ends in 8", "ends in 9", "powers of 2",
            "powers of 3", "powers of 4", "powers of 5", "powers of 6", "powers of 7",
            "powers of 8", "powers of 9", "powers of 10", "all", 
            "powers of 2 + {37}", "powers of 2 - {32}"])
print concepts


# In[88]:


data_over_hyp = np.zeros((len(concepts), 100))
for i, hyp in enumerate(concepts):
    for val in xrange(100):
        if in_hyp(hyp, val + 1):
            data_over_hyp[i,val] = 1


# In[89]:


prior = np.ones(len(concepts))
prior[0] = 5
prior[1] = 5
prior[30] = .01
prior[31] = .01
prior = prior / np.sum(prior)
print prior

# divisor is 1/size(H) where H is an element of concepts.  This is used for computing the likelihood later on
divisor = np.array([1 / np.sum(data_over_hyp[i]) for i in xrange(len(concepts))])
print divisor


# In[96]:


values = get_input_numbers()


# In[97]:


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


# In[98]:


# Get the posterior by multiplying the prior and the likelihood
posterior = np.multiply(likelihood, prior.reshape(len(concepts)))
# print posterior

# Normalize the posterior
posterior = posterior / np.sum(posterior)
print posterior


# In[99]:


predictive = np.dot(posterior, data_over_hyp)
print predictive


# In[100]:


fig = plt.figure()

fig, ax_lst = plt.subplots(3, 1, figsize=(5, 30))

y = prior.reshape(len(concepts))

ax_lst[0].barh(concepts, y)
ax_lst[0].set_xlabel("prior", fontsize=20)
ax_lst[0].set_ylabel("concepts", fontsize=20)
ax_lst[0].tick_params(labelsize=12, zorder=0)
ax_lst[0].grid(True)

ax_lst[1].barh(concepts, likelihood)
ax_lst[1].set_xlabel("likelihood", fontsize=20)
ax_lst[1].tick_params(labelsize=12)
ax_lst[1].grid(True)

ax_lst[2].barh(concepts, posterior)
ax_lst[2].set_xlabel("posterior", fontsize=20)
ax_lst[2].tick_params(labelsize=12)
ax_lst[2].grid(True)


# In[101]:


fig = plt.figure(figsize=(50, 20), dpi=80)

y = np.arange(0, 100, 1)

plt.bar(y, predictive)

plt.title('Predictive Probability', fontsize=30)

plt.ylabel('Probability', fontsize=30)
plt.tick_params(labelsize=24)
plt.xlabel('Predicted Number', fontsize=30)

plt.grid(True)
plt.show()

