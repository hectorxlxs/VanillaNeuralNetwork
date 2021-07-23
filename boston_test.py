from sklearn.datasets import load_boston
from neural_network import NeuralNetwork
import neural_network_functions as nn_funcs
from matplotlib import pyplot as plt

dataset = load_boston()['data']

"""
- CRIM     1   per capita crime rate by town
- ZN       2   proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS    3   proportion of non-retail business acres per town
- CHAS     4   Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX      5   nitric oxides concentration (parts per 10 million)
- RM       6   average number of rooms per dwelling
- AGE      7   proportion of owner-occupied units built prior to 1940
- DIS      8   weighted distances to five Boston employment centres
- RAD      9   index of accessibility to radial highways
- TAX      10  full-value property-tax rate per $10,000
- PTRATIO  11  pupil-teacher ratio by town
- B        12  1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
- LSTAT    13  % lower status of the population
- MEDV     14  Median value of owner-occupied homes in $1000's
"""

inputs = dataset[:, :12]
expected_outputs = dataset[:, 12:13]

training_data_len = int(len(inputs) * 0.7)

training_inputs = inputs[:training_data_len]
training_outputs = expected_outputs[:training_data_len]

# control_inputs = inputs[training_inputs:]
# control_outputs = expected_outputs[training_outputs:]

nn = NeuralNetwork([12, 8, 1], nn_funcs.sigmoid_act_f, nn_funcs.linear_cost_f)

nn.train(training_inputs, training_outputs, 100, 0.01)

plt.scatter(range(len(nn.historic)), nn.historic)
plt.show()