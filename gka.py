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


class Chromosome():
    def __init__(self):
        self.center = self.init_center(num_cluster)
        self.dist = np.empty(num_cluster)
        self.sol = self.random_init_sol()

        self.fitness_value = self.fitness()
        self.mutation_rate = [0.0] * len(data)
#        self.mutation()    #mutate sol string and update center

#        self.cluster()

     
    #隨機選n個中心點   
    def init_center(self, num_cluster):
#        print("hello")
#        print(len(data.iloc[1]))
        center = np.empty([num_cluster,len(data[1])])
        for i in range(num_cluster):
            random_node = int((random.randint(1,150) + random.randint(1,150)) / 2)
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
        dis = 0
        for i in range(len(a)):
            dis = dis + pow((a[i]-b[i]),2)
        return dis
    
    
    def fitness(self):
        fitness_value = 0.0
        for i in range(len(self.sol)):
            #print(self.sol[i])
#           print(fitness_value)
#           print([self.sol[i]]) 
           fitness_value = fitness_value + self.distance(data[i], self.center[self.sol[i]])
        print('fitness_value = ',fitness_value)
        return fitness_value
    
    
    def mutation(self):
        sol_distribution = sorted(self.sol)
#        print(sol_distribution)
        '''step 1 : choose the value to mutate'''
        for i in range(len(data)):
            self.mutation_rate[i] = random.random()
            if(self.mutation_rate[i] > 0.9):
#                print(i, self.mutation_rate[i])
                '''step 2 : do mutation on chosen point by wheel'''
#                print(str(self.sol[i]) + ' change to ' + str(random.choice(sol_distribution)))
                self.sol[i] = random.choice(sol_distribution)
                
        '''step 3 : update center node'''
        tmp_center = self.center
        for i in range(len(self.center)):
            count = 0
            for j in range(len(self.sol)):
                if self.sol[j] == i:
                    tmp_center[i] = tmp_center[i] + data[j]
                    count = count + 1
            self.center[i] = sum(tmp_center) / count
            
#        print('self.sol before cluster = ', self.sol)
        
    def cluster(self):
        '''step 1: calaulate the distance between each node and the center it belongs to.'''
        data2 = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
#        print('data2 = ',(data2))
        for n in range(len(data2)):
            for i in range(len(self.center)):
                self.dist[i] = self.distance(data2.iloc[n], self.center[i])
            min_idx = self.dist.argmin()
            '''step 2: updata the chromosome''' 
            self.sol[n] = min_idx        
        return self.sol
#        print('self.sol after cluster = ', self.sol)
                                                                                                                                                            
        

#選擇菁英基因 (假設40%以內)
def selection(population, fitness_arr):
    population_size = len(fitness_arr)
    chosen_rate = 0.4
    fitness_arr = np.array(fitness_arr)
    fitness_arr = fitness_arr * (-1)
#    print(fitness_arr)
    index_of_chosen = heapq.nlargest(int(len(fitness_arr)*chosen_rate), range(len(fitness_arr)), fitness_arr.take) 
#    print(index_of_chosen)
    
    '''put the value_of_chosen into array'''
    new_populations = []
    for i in range(len(index_of_chosen)):
        new_populations.append(population[index_of_chosen[i]])
        
    '''fill chosen value for full'''    
    while(len(new_populations) < population_size):
        idx = random.choice(index_of_chosen)
#        print(idx)
        new_populations.append(population[idx])
#    print(new_populations)
    return new_populations

    
def main():    
    population = [Chromosome()]*population_size
#    print(population)
    fitness_arr = [0.0]*population_size
#    print('population[0].sol = ',population[0].sol)
#    print('population[0].center = ', population[0].center)
#    print('population[0].fitness_value = ', population[0].fitness_value)
    
    for j in range(population_size):
        total_fitness_value = 0
        population[j] = Chromosome()
    
    #for each generation
    for i in range(MAX_GEN):
        print('the ', i+1, 'th iteration: ')
        population = selection(population, fitness_arr)
        
        #for each chromosome
        for j in range(population_size):
            total_fitness_value = 0
#            print('the ', j, 'th chromosome: ')
#            print(population[j].sol)
            population[j].mutation()
            population[j].cluster()
#            fitness_arr[j] = population[j].fitness()
            fitness_arr[j] = population[j].fitness_value
#            print(population[j].fitness_value)
            total_fitness_value = total_fitness_value + population[j].fitness_value
            avg_fitness_value = total_fitness_value / population_size
            print('sse = ',avg_fitness_value)
        print("============")
        
        
#    print(i, "th iteration population","\n",population[j] , "\n","=============================")
#        print(fitness_arr)
#        new_populations = selection(population, fitness_arr)
        
#        print(new_populations[j].center)
#        print(new_populations[j].sol)
        print("==============================")
    
    
    
    
    
if __name__ == "__main__":
    df = pd.read_csv('iris2.data')
    
    data = df.drop(columns = ['class']).values
#    print('data',data)
    population_size = 10
    num_cluster = 3
    MAX_GEN = 20
    main()