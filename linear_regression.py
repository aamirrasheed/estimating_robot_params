import numpy as np
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.loadtxt("traj_sensor_data.txt", delimiter=',', )
Y = np.loadtxt("traj_labels.txt", delimiter=',', )

# read in data
examples_train = []
mass_labels_train = []
force_bias_labels_train = []
mr_labels_train = []
moment_bias_labels_train = []

examples_test = []
mass_labels_test = []
force_bias_labels_test = []
mr_labels_test = []
moment_bias_labels_test = []

num_training_examples = X.shape[0] * 0.9
num_test_examples = X.shape[0] - num_training_examples
for i in range(X.shape[0]):
    # read in labels
    if i <= num_training_examples:
        mass_labels_train.append(Y[i:0])
        force_bias_labels_train.append(Y[i:1])
        mr_labels_train.append(Y[i:2])
        moment_bias_labels_train.append(Y[i:3])

        # read in training data, 6 sensor readings per row
        examples_train.append(X[i])
    else: 
        mass_labels_test.append(Y[i:0])
        force_bias_labels_test.append(Y[i:1])
        mr_labels_test.append(Y[i:2])
        moment_bias_labels_test.append(Y[i:3])

        # read in test data, 6 sensor readings per row
        examples_test.append(X[i])

# train models
mass_model = LinearRegression()
force_bias_model = LinearRegression()
mr_model = LinearRegression()
moment_bias_model = LinearRegression()

mass_model.fit(examples_train, mass_labels_train)
force_bias_model.fit(examples_train, force_bias_labels_train)
mr_model.fit(examples_train, mr_labels_train)
moment_bias_model.fit(examples_train, moment_bias_labels_train)

# output predictions and error
mass_predict = mass_model.predict(examples_test)
force_bias_predict = force_bias_model.predict(examples_test)
mr_predict = mr_model.predict(examples_test)
moment_bias_predict = moment_bias_model.predict(examples_test)

print("MSE of mass labels: ", mean_squared_error(mass_predict, mass_labels_test))
print("MSE of force bias labels: ", mean_squared_error(force_bias_predict, force_bias_labels_test))
print("MSE of mr labels: ", mean_squared_error(mr_predict, mr_labels_test))
print("MSE of moment_bias labels: ", mean_squared_error(moment_bias_predict, moment_bias_labels_test))

print("Average error of mass: ", (1/num_test_examples)*sum(np.abs(mass_predict - mass_labels_test)))
print("Average error of mr: ", (1/num_test_examples)*sum(np.abs(mr_predict,mr - labels_test)))