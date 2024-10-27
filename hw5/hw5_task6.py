#
# Template for Task 6: Random Forest Classification 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
from sklearn.ensemble import RandomForestClassifier
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
# pick five values of m by yourself 
m_values = [1, 40, 60, 80, 120]
# --- end of task --- #

er_test = []
for m in m_values: 
    # --- Your Task --- #
    # implement the random forest classification method 
    # you can directly call "RandomForestClassifier" from the scikit learn library

    # Initialize the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=m, random_state=42)

    # Train the model on the training data
    model.fit(sample_train, label_train)

    # Predict on the test data
    y_pred = model.predict(sample_test)

    # store classification error on testing data here 
    er = 1 - np.mean(label_test == y_pred)
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(m_values, er_test)
plt.xlabel('m')
plt.ylabel('Classification Error')
plt.title("Random Forest Classifier")
plt.show()

print("At the beginning (m < 10), the error is very larger. However, it seems that the error goes down from m=1 to around m=60 and then from there it seems to increase a little bit when m>60 (bounce back). So it seems that the idea value of m here is around m=60 for the lowest error term.")

