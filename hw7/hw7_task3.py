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
activation_function_list = ['identity', 'logistic', 'tanh', 'relu']

er_train_activation = []
er_test_activation  = []

for activation_func in activation_function_list: 
    
    # train a MLP classification model 
    hidden_layer_array = [80] * 5 # 80 neurons in 5 hidden layers
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_array, activation=activation_func, solver='adam', max_iter=10000, random_state=42)
    mlp.fit(sample_train, label_train)
    label_train_pred = mlp.predict(sample_train)
    label_test_pred = mlp.predict(sample_test)
    
    # evaluate training error and testing error 
    er_train = 1 - accuracy_score(label_train, label_train_pred)
    er_test = 1 - accuracy_score(label_test, label_test_pred)

    print(f"Activation Function: {activation_func} | Training Error: {er_train} | Testing Error: {er_test}")

    er_train_activation.append(er_train)
    er_test_activation.append(er_test)

print()
print("Here we can see that the list of activation functions in terms of highest to lowest training error is logistic, tanh, identity, & relu.")
print("Then the list of activation functions in terms of highest to lowest test error is logistic, relu, identity, & tanh.")
print("Here we can just tell that the logistic model is the worst one. Then the identity function seems to match up the training and testing error the best, while trying to lower the error.")
print("The relu tries to lower the training error really hard and results in a little bit of overfitting compared to the other functions. and tanh seems to get the best generalization resulting in the lowest test error.")
   
plt.figure()
plt.plot(activation_function_list,er_train_activation , label='Training Error')
plt.plot(activation_function_list,er_test_activation , label='Testing Error')
plt.xlabel('Activation Function') 
plt.ylabel('Classification Error')
plt.title('Error Rate vs Activation Function')
plt.legend()
plt.show()


