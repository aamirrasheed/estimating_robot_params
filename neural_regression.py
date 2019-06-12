import numpy as np
from io import StringIO
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
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

model = Sequential()
model.add(Dense(6, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))
model.add(Dense(150, kernel_initializer='normal',activation='relu'))
model.add(Dense(150, kernel_initializer='normal',activation='relu'))
model.add(Dense(4, kernel_initializer='normal',activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

model.fit(examples_train, mass_labels_train, epochs=500, batch_size=32, validation_split=0.2)
mass_predict = model.predict(examples_test)
print("Average error of mass: ", (1/num_test_examples)*sum(np.abs(mass_predict - mass_labels_test)))

model.fit(examples_train, mr_labels_train, epochs=500, batch_size=32, validation_split=0.2)
mr_predict = model.predict(examples_test)
print("Average error of mr: ", (1/num_test_examples)*sum(np.abs(mr_predict - force_bias_labels_test)))