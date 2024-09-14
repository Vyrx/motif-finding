#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd 
import random
import math

#f = open("E-MTAB-8626_tpms.tsv", "r")
#print(f.read())

df = pd.read_csv('E-MTAB-8626_tpms.tsv', sep='\t', skiprows=4)
df = df.dropna() # Drop missing values
#print(df)
data = df.to_numpy()
print(data)


# ## K - means

# Initialization

# In[9]:


## Only care about index 2 ~ 10 (9 values)

k = 100
data_num = data[:, 2:] # Ignore 1st and 2nd column

shape = np.shape(data_num)
sample = random.sample([i for i in range(shape[0])], k)
centroid = np.array([data_num[i] for i in sample])

belong_to_cluster = np.zeros(shape[0]) # Which cluster each data belong to
cluster = [[] for i in range(shape[0])] # Index of data in each cluster

def get_dist(x, y):
    val = 0
    if len(x) != len(y):
        raise Exception("Different Array Length")
    for i in range(len(x)):
        val += (x[i] - y[i]) * (x[i] - y[i])
    return math.sqrt(val)


# Main loop

# In[10]:


it = 0

while True:
    print("it:", str(it))
    it += 1

    ## Update cluster
    did_cluster_update = False   
    cluster = [[] for i in range(shape[0])]
    up = 0

    for i in range(shape[0]):
        min_cluster = 0
        min_dist = get_dist(data_num[i], centroid[0])
        for j in range(1, k):
            cur_dist = get_dist(data_num[i], centroid[j])
            if cur_dist < min_dist:
                min_cluster = j
                min_dist = cur_dist
        if belong_to_cluster[i] != int(min_cluster):
            did_cluster_update = True
            up += 1
        belong_to_cluster[i] = int(min_cluster)
        cluster[min_cluster].append(i)

    if did_cluster_update == False or it >= 750:
        break

    ## Update centroid  
    for i in range(k):
        if len(cluster[i]) == 0:
            continue
        new_cent = np.zeros(9)
        for index in cluster[i]:
            new_cent = new_cent + data_num[index]
        new_cent = new_cent / len(cluster[i])
        centroid[i] = new_cent

    print("Updated: " + str(up))
    
        


# In[11]:


print(data_num[0])
print(data_num[1])
print((np.zeros(9) + data_num[1])/2)
len(cluster[4])

