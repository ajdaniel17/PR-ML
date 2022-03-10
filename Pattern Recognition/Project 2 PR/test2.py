import numpy
import matplotlib.pyplot as plt
from random import seed
from random import randrange

import libsvm.svmutil as svm

seed(1)

# Creating Data (Dense)
train = list([randrange(-10, 11), randrange(-10, 11)] for i in range(10))
labels = [-1, -1, -1, 1, 1, -1, 1, 1, 1, 1]
options = '-t 0'  # linear model
print(train)
print(labels)
# Training Model
model = svm.svm_train(labels, train, options)


# Line Parameters
w = numpy.matmul(numpy.array(train)[numpy.array(model.get_sv_indices()) - 1].T, model.get_sv_coef())
b = -model.rho.contents.value
if model.get_labels()[1] == -1:  # No idea here but it should be done :|
    w = -w
    b = -b

print(w)
print(b)

# Plotting
plt.figure(figsize=(6, 6))
for i in model.get_sv_indices():
    plt.scatter(train[i - 1][0], train[i - 1][1], color='red', s=80)
train = numpy.array(train).T
plt.scatter(train[0], train[1], c=labels)
plt.plot([-5, 5], [-(-5 * w[0] + b) / w[1], -(5 * w[0] + b) / w[1]])
plt.xlim([-13, 13])
plt.ylim([-13, 13])
plt.show()