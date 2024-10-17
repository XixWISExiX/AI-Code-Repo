#
# Template for Task 2: Logistic Regression 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
# .......
# --- end of task --- #

# -------------------------------------
# load data 
data = np.loadtxt('../data_sets/diabetes.csv', delimiter=',')
[n,p] = np.shape(data)
# 75% for training, 25% for testing 
num_train = int(0.75*n)
num_test = int(0.25*n)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]
# -------------------------------------


# --- Your Task --- #
# pick a proper number of iterations 
num_iter = 1000
# randomly initialize your w 
w = [0] * (p-1)
alpha = 0.01
# --- end of task --- #

er_test = []


# --- Your Task --- #
# implement the iterative learning algorithm for w
# at the end of each iteration, evaluate the updated w 
for iter in range(num_iter): 

    ## update w
    y_hat = 1 / (1 + np.exp(-1 * np.dot(sample_train, w)))  # Prediction
    error = label_train - y_hat      # Error
    w = w + alpha * np.dot((label_train - y_hat) * y_hat * (1 - y_hat), sample_train) # w update

    ## evaluate testing error of the updated w 
    er = 1 - np.mean(label_train == np.where(y_hat >= 0.5, 1, 0))
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(er_test)
plt.xlabel('Iteration')
plt.ylabel('Classification Error')
plt.title('Logistic Regression')
plt.show()
