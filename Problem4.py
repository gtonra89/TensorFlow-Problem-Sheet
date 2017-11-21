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

# 1. Load Iris species dataset
# Split into input (X) and output (Y) variables
Datacsv = pd.read_csv("Iris.csv")

print(Datacsv)
