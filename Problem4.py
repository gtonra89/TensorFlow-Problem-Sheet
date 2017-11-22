##added dependencies

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import sys

# Fix random seed for reproducibility
np.random.seed(7)

# Load Iris species dataset
# Split into input (X) and output (Y) variables
Datacsv = pd.read_csv("Iris.csv")

print(Datacsv)


# Delete the first column which is "Id"
Datacsv = Datacsv.drop("Id", 1)

# Convert Species to numeric id
Datacsv["Species"] = Datacsv["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).astype(int)

# Converts data to anp matrix
dataset = Datacsv.as_matrix()



# Preparing Cross Validation

trainingSet = np.random.randint(dataset.shape[0], size=100)
testSet = np.random.randint(dataset.shape[0], size=45)

xTrain = dataset[trainingSet, :4]
yTrain = to_categorical(dataset[trainingSet, 4], num_classes=3)

xTest = dataset[testSet, :4]
yTest = to_categorical(dataset[testSet, 4], num_classes=3)


# Define Model

model = Sequential()  # Creating Sequential Model
model.add(Dense(32, input_dim=4, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(3, activation="sigmoid"))

# Compile Model
# sgd = Stochastic gradient descent optimizer

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)


# Fit Model

model.fit(
    xTrain,
    yTrain,
    epochs=105,
    batch_size=10
)

# Evaluate Model

scores = model.evaluate(xTest, yTest)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Save predictions

predictions = pd.DataFrame({
    "Ids": range(1, 151),
    "Species": [p.argmax() for p in model.predict(dataset[:, :4])]
})

predictions["Species"] = Datacsv["Species"].map({
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}).astype(str)

predictions.to_csv("predictions.csv", index=False)