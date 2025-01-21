import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Concatenate, Lambda, Dense, Dropout, Activation, Flatten, LSTM, SpatialDropout2D, Conv2D, MaxPooling2D,
    AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, BatchNormalization, ReLU, Input
)
from tensorflow.keras import optimizers, regularizers, backend as K
from tensorflow.keras.models import Sequential, Model, load_model

from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, classification_report,
    cohen_kappa_score, hamming_loss, log_loss, zero_one_loss, matthews_corrcoef, roc_curve, auc
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tensorflow.keras.utils import to_categorical, plot_model

from time import time
import time as tm
import datetime

from skimage.util.shape import view_as_blocks
from contextlib import redirect_stdout

import cv2
import glob
import os
import random
import ntpath
import copy

from tfkan.layers import DenseKAN

import mlflow
from mlflow.tensorflow import MlflowCallback