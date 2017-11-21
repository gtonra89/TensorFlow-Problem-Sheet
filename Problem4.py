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
