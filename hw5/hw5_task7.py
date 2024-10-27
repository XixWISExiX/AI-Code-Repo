#
# Template for Task 7: Fairness in machine learning 
# 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --- Your Task --- #
# import libraries as needed 
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
# --- end of task --- #

# ----------------------------------------------------------
# We will experiment on the student performance data set
# You can find description of the original data set here
# https://archive.ics.uci.edu/dataset/320/student+performance
# We provide a preprocessed data set "student.csv". 
# The 1st column contains gender (1 for female; 0 for male)
# The last column contains final score (we will binarize it)
# ----------------------------------------------------------
data = np.loadtxt('../data_sets/student.csv', delimiter=',')
# now we binarize the label so 1 means > 10 and 0 means <= 10
data[data[:,-1]<=12,-1] = 0
data[data[:,-1]>0,-1] = 1
#
[n,p] = np.shape(data)
# 75% for training, 25% for testing 
num_train = int(0.75*n)
num_test = int(0.25*n)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]
# -------------------------------------


# --- Baseline Method --- #
# standard training and testing 
model = LinearDiscriminantAnalysis()
model.fit(sample_train,label_train)
label_pred = model.predict(sample_test)
# now, we need to separately evaluate the 
# error on male students and female students
idx_female = np.where(sample_test[:,0]==1)[0]
idx_male = np.where(sample_test[:,0]==0)[0]
er_female = 1-accuracy_score(label_test[idx_female],label_pred[idx_female])
er_male = 1-accuracy_score(label_test[idx_male],label_pred[idx_male])
er_gap = abs(er_male - er_female)
# you will see, error is 10% higher on male than female 
print("Standard Method")
print('Group 1', er_male)
print('Group 2', er_female)
print('Error Gap', er_gap)

# --- Your Task --- #
# now, implement whatever method you choose 
# aim to reduce the gap between two errors 
model = LinearDiscriminantAnalysis()

smote = SMOTE(random_state=44)
sample_train, label_train = smote.fit_resample(sample_train, label_train)

model.fit(sample_train, label_train)
# whatever method you have, store your model prediction 
# on testing set in "label_pred". Then run the following code
label_pred = model.predict(sample_test)
idx_female = np.where(sample_test[:,0]==1)[0]
idx_male = np.where(sample_test[:,0]==0)[0]
er_female = 1-accuracy_score(label_test[idx_female],label_pred[idx_female])
er_male = 1-accuracy_score(label_test[idx_male],label_pred[idx_male])
er_gap = abs(er_male - er_female)
print("Fair Method")
print('Group 1', er_male)
print('Group 2', er_female)
print('Error Gap', er_gap)
# --- end of task --- #


print('Here we can see that there is around a 10% gap between the male error (Group 1, 38% error) vs female error (Group 2 29% error). In the basic logistic regression. However in my method (Implement SMOTE with logistic regression) the error of both classes drops to 27% and the gap error shrinks to 0.3% which gives us not just an equal prediciton for both classes, but an equal chance that the prediction is wrong too, which will reduce any unfairness going on compared to the previous implementation.')