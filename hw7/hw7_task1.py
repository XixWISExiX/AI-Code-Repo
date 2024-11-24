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
k_values = [1,10,40,80,160]

er_train_k = []
er_test_k = []
for k in k_values: 
    
    # train a MLP classification model 
    hidden_layer_array = [k] * 5 # k neurons in 5 hidden layers
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_array, activation='relu', solver='adam', max_iter=10000, random_state=42)
    mlp.fit(sample_train, label_train)
    label_train_pred = mlp.predict(sample_train)
    label_test_pred = mlp.predict(sample_test)
    
    # evaluate training error and testing error 
    er_train = 1 - accuracy_score(label_train, label_train_pred)
    er_test = 1 - accuracy_score(label_test, label_test_pred)
    er_train_k.append(er_train)
    er_test_k.append(er_test)

print("At first the model is underfit with a high error rate in both training and testing.")
print("Then at around 80 neurons the model looks to be the most correctly fit, resulting in a lower testing error rate")
print("Finally after the 80 neurons, the 160 neurons the model error seems to be growing again while the training error is going down meaning the model is being overfit.")
   
plt.figure()
plt.plot(k_values,er_train_k, label='Training Error')
plt.plot(k_values,er_test_k, label='Testing Error')
plt.xlabel('k neurons per hidden layer (5)')
plt.ylabel('Classification Error')
plt.title('Error Rate vs # of neurons per hidden layer')
plt.legend()
plt.show()


