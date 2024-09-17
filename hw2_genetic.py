###
# Part 1
###
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('data_sets/CreditCard.csv')

# Drop rows with any `null` values or empty strings
df = df.replace('', pd.NA).dropna()
df = df.reset_index(drop=True)

# Encode Values In Data Sets
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
df['CarOwner'] = df['CarOwner'].map({'Y': 1, 'N': 0})
df['PropertyOwner'] = df['PropertyOwner'].map({'Y': 1, 'N': 0})

# Drop the Ind_ID
df = df.drop(columns=['Ind_ID'])

# Display the first few rows of the DataFrame
print(df.head())
print('Number of rows =',len(df))

###
# Part 2
###

# Get X and y values
y = df['CreditApprove']
X = df.drop(columns=['CreditApprove']) 

###
# Part 3
###

# Init w
w = [-1, -1, -1, -1, -1, -1]

# Init w'
w_prime = [1, 1, 1, 1, 1, 1]

# Genetic Search General Code
# def evolve(fitness function):
#     population = 1000
#     evaluation = evaluate(population, fitness)
#     while(evaluation < goal and time left to run):
#         new_population = set([])
#         while(new_population is not finished):
#             select individuals from population
#             crossover & mutate
#         population = new_population
#         evaluation = evaluate(population, fitness)
#     return best current solution


############################################################################
# TASK 2
############################################################################

def find_er_w(w):
    fx = [0] * len(X)
    er_w_sum = 0
    for i in range(len(X)):
        for j in range(len(X.columns)):
            fx[i] += w[j] * X.iloc[i,j]
        er_w_sum += (fx[i] - y[i])**2
    er_w = er_w_sum  / len(X)
    return er_w

# TODO Task 2
# fitness function = e^(-er(w))
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import random

def fitness(w):
    return math.exp(-find_er_w(w))

def genetic_search(w, population_size=4, threshold=1000):
    population = np.random.choice([-1,1], size=(population_size, len(X))) # Generate a random population
    # print(population)
    for generation in range(threshold):
        fitnesses = np.array([fitness(w) for w in population])
        # print(fitnesses)
        new_population = []

        for _ in range(population_size // 2):
            # print(sum(fitnesses))
            probabilities = 1 - (fitnesses / sum(fitnesses)) # Lower error = higher probability
            # print(sum(probabilites))
            parent1 = random.choices(population, weights=probabilities, k=1)[0]
            parent2 = random.choices(population, weights=probabilities, k=1)[0]
            new_population.append(parent1)
            new_population.append(parent2)

            # TODO implement crossover
            # TODO implement mutation
        population = new_population
        

    return 'NOT DONE'

genetic_search(w)