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

# Local Search General Code
def minimize_er_w(w, w_prime):
    while True:
        fx = [0] * len(X)
        fxprime = [0] * len(X)
        er_w_sum = 0
        er_w_prime_sum = 0
        for i in range(len(X)):
            for j in range(len(X.columns)):
                fx[i] += w[j] * X.iloc[i,j]
                fxprime[i] += w_prime[j] * X.iloc[i,j]
            er_w_sum += (fx[i] - y[i])**2
            er_w_prime_sum += (fxprime[i] - y[i])**2
        er_w = er_w_sum  / len(X)
        er_w_prime = er_w_prime_sum  / len(X)
        if(er_w_prime < er_w):
            w = w_prime
        else:
            break
    print('Optimized w (general local search) =',w)

minimize_er_w(w, w_prime)

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

def genetic_search(w, threshold=1000):
    return 'a'