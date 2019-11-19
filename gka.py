#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:00:17 2018

@author: snoopyknight
"""

import pandas as pd
import numpy as np
import random
import math
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import csv

class Chromosome():
    def __init__(self):
        self.center = self.init_center(num_cluster)
        self.dist = np.empty(num_cluster)
        self.sol = self.random_init_sol()
        self.mutation_rate = [0.0] * len(data)
#        self.mutation()    #mutate sol string and update center

#        self.cluster()

     
    #隨機選n個中心點   
    def init_center(self, num_cluster):
#        print("hello")
#        print(len(data.iloc[1]))
        center = np.empty([num_cluster,len(data[1])])
        for i in range(num_cluster):
            random_node = int((random.randint(0,149) + random.randint(0,149)) / 2)
            center[i] = data[random_node]            
        return(center)
    
    
    #對於每一個染色體，隨機產生一組解
    def random_init_sol(self):
        sol = [0]*len(data)
        for i in range(len(data)):
            sol[i] = np.random.randint(0,3)
#        print((sol))
        return sol
         
            
    def distance(self, a, b):
        tmp = 0
        for i in range(len(a)):
            tmp = tmp + pow((a[i]-b[i]),2)
        dis = math.sqrt(tmp)
        return dis

    
    def mutation(self):
        mutation_threshold = 0.8
#        print("Before mutation : ", self.sol)
        sol_distribution = sorted(self.sol)
#        print(sol_distribution)
        '''step 1 : choose the value to mutate'''
        for i in range(len(data)):
            self.mutation_rate[i] = random.random()
            if(self.mutation_rate[i] > mutation_threshold):
#                print(i, self.mutation_rate[i])
                '''step 2 : do mutation on chosen point by randomly selected from sol(1~K) accroding to the distribution'''
#                print(str(self.sol[i]) + ' change to ' + str(random.choice(sol_distribution)))
                self.sol[i] = random.choice(sol_distribution)                
#        print("After mutation : ", self.sol)
        return self.sol


    
    def cluster(self):
        '''step 1: calculate center node'''
        for i in range(len(self.center)):
            count = 0
            for j in range(len(self.sol)):
                if self.sol[j] == i:
                    self.center[i] = self.center[i] + data[j]
                    count = count + 1
            if count == 0:   #To prevent inf
                self.center[i] = self.center[i]
            else:
                self.center[i] = (self.center[i]) / count
        
        '''step 2: reassign each data point to the cluster with the nearest cluster center'''
#        nodes = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])        
#        nodes = pd.DataFrame(data)
        nodes = features
        for n in range(len(nodes)):
            for i in range(len(self.center)):
                self.dist[i] = self.distance(nodes.iloc[n], self.center[i])
            min_idx = self.dist.argmin()  #choose the nearest center
            self.sol[n] = min_idx    #reassign each data point belongs to  
        return self.sol, self.center
    


    
def fitness_value(population):
    sse = [0.0]*len(population)
    fsw = [0.0]*len(population)
    c = 2  #constant between 1~3
    
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
#    print("sse = ", sse)   
    return fitness, sse


        

#選擇菁英基因 (假設68%以內)
def selection(population, fitness_arr):
    population_size = len(fitness_arr)
    fitness_arr = np.array(fitness_arr)
#    total_candidate_chromosome_size = int(sum(fitness_arr))
    each_candidate_chromosome_size = [0.0]*len(fitness_arr)
    
    '''========roulette wheel selection ========='''
    '''Step 1 : Generate all of candidate_chromosome by the fitness value'''
    for i in range(len(fitness_arr)):
        each_candidate_chromosome_size[i] = int(fitness_arr[i])
        
    candidate_chromosome = list()
    for i in range(population_size):
        cnt = 0
        while cnt < each_candidate_chromosome_size[i]:
            candidate_chromosome.append(population[i])
            cnt+=1
#            print("candidate_chromosome.sol = ", candidate_chromosome[i].sol)
#            print("candidate_chromosome.center = ", candidate_chromosome[i].center) 
            
    '''Step 2: Random choice from all of candidate_chromosome'''
    new_populations = random.choices(candidate_chromosome, k=population_size)

    return new_populations
    
 
    


def main():    
    population = [None]*population_size    #initial population
    '''initial each Chromosome with random sol'''
    for i in range(population_size):
        population[i] = Chromosome()
        #print("population[",i,"].sol",population[i].sol)
#        print("population[",i,"].center",population[i].center)

    geno = MAX_GEN    
    final_population = list()
    #for each generation
    while(geno > 0):   
        print('the ', MAX_GEN - geno + 1, 'th iteration: ')           
            
            
        '''calculate fitness values of each string''' 
        fitness_arr = fitness_value(population)
#        print("fitness = ",fitness_arr)    
            
        '''selection'''
        #If fitness_value is convergence, selection function can not be called, because it will return IndexError.
        try:
            population = selection(population, fitness_arr[0])
        except IndexError:
            print("======= Iteration End !! ========")
            break
        
        new_population = list()
        '''mutation and K-means operator'''
        for i in range(population_size):
            mutate = population[i].mutation()
            KMO = population[i].cluster()
#            print('population[', i ,'].cluster()', KMO[0])
#            print('population[', i ,'].center()', KMO[1])
            new_population.append(KMO)    
        
        geno = geno-1  
        print("============")
        
    final_population = new_population    
    
    chosen_index = np.argmax(fitness_arr[0])
#    print("best chromosome = ",final_population[chosen_index])
#    print("best SSE = ",fitness_arr[1][chosen_index])
    
    with open('GKA result.csv','a',newline='') as fd:
        fdWrite = csv.writer(fd)
#        fdWrite.writerow([fitness_arr[1][chosen_index],final_population[chosen_index][0],final_population[chosen_index][1]])
        fdWrite.writerow([fitness_arr[1][chosen_index]])

    
    
    
    
if __name__ == "__main__":
    #prepare data
    df = pd.read_csv('iris.data', header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    for i in range(len(df['class'])):
        if df['class'].iloc[i] == 'Iris-setosa':
            df['class'].iloc[i] = 0
        elif df['class'].iloc[i] == 'Iris-versicolor':
            df['class'].iloc[i] = 1
        else:
            df['class'].iloc[i] = 2

#    print(df.head())
    features = df.drop(columns = ['class'])
    labels = df["class"]
    data = df.drop(columns = ['class']).values
    real_sol = df["class"].values
#    print('data',data)
    global population_size 
    population_size = 200
    num_cluster = 3
    MAX_GEN = 20  
    main()
