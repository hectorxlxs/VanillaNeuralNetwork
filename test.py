import neural_network_functions as nn_funcs
from neural_network import NeuralNetwork
from matplotlib import pyplot as plt
import random

nn = NeuralNetwork([3, 1], nn_funcs.sigmoid_act_f, nn_funcs.linear_cost_f)

# DATASET -> I'm trying to create a nn that detects if sum of 3 inputs are 1.5 or higher

dataset = []
expected_outputs = []

for i in range(2000):
    dataset.append([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])

    if dataset[i][0] > 0.8 or dataset[i][1] > 0.3 or dataset[i][2] > 0.3:
        expected_outputs.append([1])
    else:
        expected_outputs.append([0])

    # expected_outputs.append([1 if sum(dataset[i]) >= 1.5 else 0])

# Dividing data in two groups, training group and control group
training_data_len = int(len(dataset) * 0.7)

training_data = dataset[:training_data_len]
expected_training_data = expected_outputs[:training_data_len]

control_data = dataset[training_data_len:]
expected_control_data = expected_outputs[training_data_len:]

nn.train(training_data, expected_outputs[:training_data_len], 200, 0.01)

print(nn)

correct_predictions = 0
for data_index in range(len(control_data)):
    output = nn.get_output(control_data[data_index])
    if output[0] > 0.5:
        output = 1
    else:
        output = 0

    if output == expected_control_data[data_index][0]:
        correct_predictions += 1

print("Predicted correctly {}/{} inputs".format(correct_predictions, len(control_data)))

X = range(len(nn.historic))
Y = nn.historic
plt.scatter(X, Y)

plt.show()

print("Prueba introduciendo números del 0 al 1")

while True:
    _input = [float(input("x0: ")), float(input("x1: ")), float(input("x2: "))]
    print(nn.get_output(_input))









