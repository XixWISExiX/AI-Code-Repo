import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = np.loadtxt('../data_sets/diabetes.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# training data 
num_train = int(0.5*n)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# testing data 
num_test = int(0.5*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# ----------------------- #
# --- Hyper-Parameter --- #
# ----------------------- #
m_values = [1,3,5,15,25]

er_train_m = []
er_test_m = []
for m in m_values: 
    
    # train a MLP classification model 
    hidden_layer_array = [10] * m # 7 neurons in m hidden layers
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_array, activation='relu', solver='adam', max_iter=10000, random_state=42)
    mlp.fit(sample_train, label_train)
    label_train_pred = mlp.predict(sample_train)
    label_test_pred = mlp.predict(sample_test)
    
    # evaluate training error and testing error 
    er_train = 1 - accuracy_score(label_train, label_train_pred)
    er_test = 1 - accuracy_score(label_test, label_test_pred)
    er_train_m.append(er_train)
    er_test_m.append(er_test)
    
print("At 1 hidden layer the error of the test set seems to be at it's smallest.")
print("Then up to 15 hidden layers the training error goes down, but the testing error goes up which shows overfitting.")
print("Finally at 25 hidden layers the training error sky rockets and so does the testing error, which shows a potential underfitting that's worse than the base case.")
   
plt.figure()
plt.plot(m_values,er_train_m, label='Training Error')
plt.plot(m_values,er_test_m, label='Testing Error')
plt.xlabel('10 neurons per hidden layer (m)') # need to change it to "m" value for figure 2
plt.ylabel('Classification Error')
plt.title('Error Rate vs 10 neurons per (m) hidden layer')
plt.legend()
plt.show()


