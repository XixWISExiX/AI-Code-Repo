###
# Part 1
###
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('data_sets/CreditCard.csv')

# Drop rows with any `null` values or empty strings
df = df.replace('', pd.NA).dropna()

# Encode Values In Data Sets
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
df['CarOwner'] = df['CarOwner'].map({'Y': 1, 'N': 0})
df['PropertyOwner'] = df['PropertyOwner'].map({'Y': 1, 'N': 0})

# Drop the Ind_ID
df = df.drop(columns=['Ind_ID'])

# Display the first few rows of the DataFrame
print(df.head())
print(len(df))

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

# Init w error
er_w = [0] * 6


print(X.iloc[140,1])
for j in range(len(X.columns)):
    local_sum = 0
    for i in range(len(X)):
        print(i, j, X.iloc[i,j], w[j], y[i])
        local_sum += (w[j]*X.iloc[i,j] - y[i])**2
    er_w[j] = 1/len(X) * local_sum

print(er_w)
