# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:05:50 2019

@author: snoopyknight
"""


import pandas as pd
import numpy as np
import random
import math
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import csv

    
def SSE(data, center):
    sse = [0.0]*len(population)
    fsw = [0.0]*len(population)
    c = 1  #constant between 1~3
    
    #calculate sse of each chromosome
    #sse: sum of distance of each point betwwen the within cluster
    for i in range(len(population)):
       for j in range(len(data)):
           sse[i] = sse[i] + population[i].distance(data[j], population[i].center[population[i].sol[j]])
       fsw[i] = (-1) * sse[i]
       
    #calculate average of fsw 
    avg_fsw = sum(fsw)/len(sse)
    #calculate standard deviation of f_sw
    sigma = np.std(fsw, dtype=np.float64)
    
    #calculate fitness value 
    fitness = [0.0]*len(population)
    for i in range (len(population)):
        fitness[i] = fsw[i] - (avg_fsw - c*sigma)    #formula of fitness value
        if fitness[i] >= 0:
            pass
        else:
            fitness[i] = 0
    print("sse = ", sse)   
    return fitness, sse



def load_data(filename):
    df = pd.read_csv(filename, header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    for i in range(len(df['class'])):
        if df['class'].iloc[i] == 'Iris-setosa':
            df['class'].iloc[i] = 0
        elif df['class'].iloc[i] == 'Iris-versicolor':
            df['class'].iloc[i] = 1
        else:
            df['class'].iloc[i] = 2
    return df 


def init_center(data, num_cluster):
    center = np.zeros([num_cluster,len(data[0])])
    for i in range(num_cluster):
        random_node = int((random.randint(0,149) + random.randint(0,149)) / 2)
        center[i] = data[random_node]
    return center


def distance(a, b):
    tmp = 0
    for i in range(len(a)):
        tmp = tmp + pow((a[i]-b[i]),2)
    dis = math.sqrt(tmp)
    return dis
    

def cluster(node, center):
    dist = np.zeros(len(center))
    sol = [0.0]*len(node)
    for n in range(len(node)):
        for i in range(len(dist)):
            dist[i] = distance(node[n], center[i])
        within_cluster = dist.argmin()
        sol[n] = within_cluster
    return sol

def update_center(node, sol, center):
    for i in range(len(center)):
        cnt = 0
        for n in range(len(node)):
            if sol[n] == i:      #i: within class
                center[i] += node[n]
                cnt+=1
        if cnt == 0:   #To prevent inf
            center[i] = center[i]
        else:
            center[i] = center[i] / cnt
    return center


def SSE(node, sol, center):
    sse = 0.0    
    #sse: sum of distance of each point betwwen the within cluster
    for n in range(len(node)):
        sse = sse + distance(node[n], center[sol[n]])  
    return sse

def main(): 
    data = load_data('iris.data') 
    features = data.drop(columns = ['class']).values
#    print(data.head())
    old_center = init_center(features, 3)
    geno = MAX_GEN
    while(geno > 0):   
        print('the ', MAX_GEN - geno + 1, 'th iteration: ')           
        sol = cluster(features, old_center)
        sse = SSE(features, sol, old_center)
        print("sse = ", sse)
        new_center = update_center(features, sol, old_center)
#        print("old_center", old_center)
#        print("new_center", new_center)
        old_center = new_center
        print(sol)
    
        geno = geno-1
        print("==============")
        
    with open('K-Means result.csv','a',newline='') as fd:
        fdWrite = csv.writer(fd)
        fdWrite.writerow([sse])
if __name__ == "__main__":
    num_cluster = 3
    MAX_GEN = 20
    for i in range(100):
        main()