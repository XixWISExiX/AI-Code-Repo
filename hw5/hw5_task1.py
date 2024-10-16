#
# Template for Task 1: Linear Regression 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
# .......
# --- end of task --- #

# -------------------------------------
# load data 
data = np.loadtxt('../data_sets/student.csv', delimiter=',')
[n,p] = np.shape(data)
# 75% for training, 25% for testing 
num_train = int(0.75*n)
num_test = int(0.25*n)
sample_train = np.array(data[0:num_train,0:-1])
label_train = np.array(data[0:num_train,-1])
sample_test = np.array(data[n-num_test:,0:-1])
label_test = np.array(data[n-num_test:,-1])
# -------------------------------------


# --- Your Task --- #
# pick a proper number of iterations 
num_iter = 15
# randomly initialize your w 
w = np.array([0] * (p-1)) # must be small because our method only goes up.
# --- end of task --- #

er_test = []
alpha = 0.003


# --- Your Task --- #
# implement the iterative learning algorithm for w
# at the end of each iteration, evaluate the updated w 
for i in range(num_iter): 

    ## update w
    y_hat = np.dot(sample_train, w)
    w = w + alpha * np.abs(np.sum(y_hat - label_train)) / len(label_train)


    ## evaluate testing error of the updated w 
    # we should measure mean-square-error here
    er = np.sum((y_hat -  label_train) ** 2) / len(label_train)
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(er_test)
plt.xlabel('Iteration')
plt.ylabel('Classification Error')
plt.title('Linear Regression')
plt.show()

print('For this implementation of linear regression i did not use derivaties via the video over this section for this class.')
print('So instead i used the sum of the errors divided by the number of terms times some small number alpha plus w.')
print('That was the main method for updating the w values and then the R^2 for evaluating the results.')
print('Of which we can see the results decreasing the error term with each given iteration starting from an R^2 of 130 to an R^2 of around 25')
print('in around 14 iterations.')


