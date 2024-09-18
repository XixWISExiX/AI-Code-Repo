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

import numpy as np

# Like find_er_w in hw2_local.py, just faster because numpy
def find_er_w(w):
    X_np = X.to_numpy() 
    y_np = y.to_numpy()
    
    # Calculate fx = X * w
    fx = np.dot(X_np, w)
    
    # Calculate er_w_sum = (fx - y)^2 summed over all samples
    er_w_sum = np.sum((fx - y_np) ** 2)
    
    # Calculate er_w
    er_w = er_w_sum / len(X_np)
    
    return er_w


print(find_er_w(w))
print('---')

import math
import matplotlib.pyplot as plt
import random

def fitness(w):
    return math.exp(-find_er_w(w))

def genetic_search(population_size=8, threshold=500):
    population = np.random.choice([-1,1], size=(population_size, len(X.columns))) # Generate a random population
    rounds = []
    er_w_trasformations = []
    # TODO find a way to store min er_w
    # prev_fitnesses = [10*population_size]
    for generation in range(threshold):
        fitnesses = np.array([fitness(w) for w in population])

        # if sum(prev_fitnesses) < sum(fitnesses):
            # print(sum(prev_fitnesses), sum(fitnesses))
            # break
        
        prev_fitnesses = fitnesses
        rounds.append(generation)

        new_population = []

        for chromosome in range(population_size // 2):
            probabilities = (1-fitnesses) / sum(1-fitnesses) # Lower error = higher probability
            # probabilities = (fitnesses / sum(fitnesses)) # Lower error = higher probability

            # print("fit",fitnesses)
            # print("prob",probabilities)
            # print("prob sum",sum(probabilities))

            parent1 = random.choices(population, weights=probabilities, k=1)[0]
            parent2 = random.choices(population, weights=probabilities, k=1)[0]
            new_population.append(parent1)
            new_population.append(parent2)

            # Crossover
            crossover = np.append(parent1[:3], parent2[3:])

            # Mutation change 1 random variable in each crossover
            random_index = np.random.randint(0, len(crossover))
            crossover[random_index] *= -1

            print("Generation:", generation, "Chromosome", chromosome, crossover)
            new_population.append(crossover)

        population = new_population
        generation_er_ws = np.array([find_er_w(w) for w in population])
        # w_primes = np.array([fitness(w) for w in population])
        # er_w_trasformations.append(sum(w_primes)/population_size)
        er_w_trasformations.append(min(generation_er_ws))

    w_primes = np.array([find_er_w(w) for w in population])
    print("Current w's in population", w_primes)





    plt.figure(figsize=(8,6))
    plt.plot(rounds, er_w_trasformations, marker='o', linestyle='-', color='b', label='Error')
    plt.xlabel('Round of Search')
    plt.ylabel('Error (er_w)')
    plt.title('Generation Search, Error vs. Round')
    plt.show()
        

    return 'NOT DONE'

genetic_search()
# print(find_er_w([1,-1,1,-1,1,1]))

#----------------------------------------------------------------------------------------------------------

# # Compare all values
# import itertools
# import numpy as np

# # Original array for reference
# arr = [1, 1, 1, 1, 1, 1]

# # Generate all permutations (combinations) of length 6 with values either 1 or -1
# permutations = list(itertools.product([1, -1], repeat=len(arr)))

# # Convert to a NumPy array if needed
# permutations_array = np.array(permutations)

# print(permutations_array)
# print("--------------------------------------------")

# min_val = 10
# for perm in permutations_array:
#     er_w = find_er_w(perm)
#     print(er_w)
#     min_val = min(min_val, er_w)
# print("min",min_val)