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
num_iter = 200
# randomly initialize your w 
w = [0] * (p-1)
alpha = 0.05
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
    y_hat_test = 1 / (1 + np.exp(-1 * np.dot(sample_test, w)))  # Test Prediction
    er = 1 - np.mean(label_test == np.where(y_hat_test >= 0.5, 1, 0))
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(er_test)
plt.xlabel('Iteration')
plt.ylabel('Classification Error')
plt.title('Logistic Regression')
plt.show()

print("For the update function, I used the method given in class. The only thing that could be different here is that i'm using numpy dot product instead of for loops to update the weights.")
print("We can see that the error of the approximated function drops from around 38% to 32% & then takes a around 150 iterations to finally hit the lowest error rate (32%) without any other methods (like normalization).")