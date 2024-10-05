import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import Ridge
# --- end of task --- #

# load a data set for regression
# in array "data", each row represents a community 
# each column represents an attribute of community 
# last column is the continuous label of crime rate in the community
# data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)
data = np.loadtxt('../data_sets/crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# --- Your Task --- #
# now, pick the percentage of data used for training 
# remember we should be able to observe overfitting with this pick 
# note: maximum percentage is 0.75 
per = 0.5
num_train = int(n*per)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# --- end of task --- #


# --- Your Task --- #
# We will use a regression model called Ridge. 
# This model has a hyper-parameter alpha. Larger alpha means simpler model. 
# Pick 5 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
alpha_vec = [0.1, 0.5, 1, 5, 10]
# --- end of task --- #

# er_train_alpha = []
# er_test_alpha = []
er_valid_alpha = []
for alpha in alpha_vec: 

    # pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    
    # --- Your Task --- #
    # now implement k-fold cross validation 
    # on the training set (which means splitting 
    # training set into k-folds) to get the 
    # validation error for each candidate alpha value 
    # store it in "er_valid"
    k = 5
    fold_size = len(sample_train) // k
    scores = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        X_test, y_test = sample_train[start:end], label_train[start:end]
        
        X_train = np.concatenate([sample_train[:start], sample_train[end:]], axis=0)
        y_train = np.concatenate([label_train[:start], label_train[end:]], axis=0)

        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        scores.append(score)
    mean_score = np.mean(scores)

    er_valid_alpha.append(mean_score)
print("alpha values:", alpha_vec)
print("error validation values:", er_valid_alpha)
print("Higher Score (R^2) gives you best guess, in this case it's alpha = 1.")
    # --- end of task --- #


# Now you should have obtained a validation error for each alpha value 
# In the homework, you just need to report these values

# The following practice is only for your own learning purpose.
# Compare the candidate values and pick the alpha that gives the smallest error 
# set it to "alpha_opt"
alpha_opt = alpha_vec[np.argmax(er_valid_alpha)]

# now retrain your model on the entire training set using alpha_opt 
# then evaluate your model on the testing set 
model = Ridge(alpha = alpha_opt)
model.fit(sample_train, label_train)

print("Model R^2 value on test set is:", model.score(sample_test, label_test))
