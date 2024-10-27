#
# Template for Task 4: kNN Classification 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
from scipy.spatial import distance
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
# pick five values of k by yourself 
k_values = [1, 3, 5, 7, 9]
# --- end of task --- #

er_test = []
for k in k_values: 
    # --- Your Task --- #
    # implement the kNN classification method 
    y_hat = []

    # Loop through each test sample
    for test_sample in sample_test:
        # Calculate distances between the test sample and all training samples
        distances = distance.cdist([test_sample], sample_train, 'euclidean')[0]
        
        # Find the indices of the k nearest neighbors
        nearest_neighbors_idxs = np.argsort(distances)[:k]
        # print(nearest_neighbors_idxs)
        
        # Get the labels of the k nearest neighbors
        nearest_labels = label_train[nearest_neighbors_idxs]
        
        # Predict the label (majority vote)
        prediction = np.round(np.mean(nearest_labels))
        y_hat.append(prediction)

    # store classification error on testing data here 
    y_hat = np.array(y_hat)
    er = 1 - np.mean(label_test == np.where(y_hat >= 0.5, 1, 0))
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(k_values, er_test)
plt.xlabel('k')
plt.ylabel('Classification Error')
plt.title('K Nearest Neighbors')
plt.show()


print("The algorithm above uses numpy for finding the closest values, along with a distance library to easily calculate distance. Overall this k nearest neighbors was way easier to implement in comparision to k means mostly due to the lack of centriods.")
print("At k=1 and k=2 we can tell that the error is to large and likely an overfit because we are finding the immediate values closest to the point. By k=3 this improves and by k=5 we get our lowest error. When k>5 the error seems to increase again lead me to assume that the model is now more on the underfitting side (averge to the whole data set is not useful for example).")
