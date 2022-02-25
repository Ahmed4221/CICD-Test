import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from model import *
from constants import *
import subprocess
import os
print(tf.__version__)
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


def getData():
    url = DATA_URL
    raw_dataset = pd.read_csv(url, names=DATA_COLUMNS,
                            na_values='?', comment='\t',
                            sep=' ', skipinitialspace=True)
    return raw_dataset

def preprocessData(dataset):
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
    return dataset

def splitTraintTest(dataset):
    dataset = dataset[FEATURES_TO_USE]
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    return train_dataset,test_dataset

def getNormalizer(input_shape_incoming = None):
    if input_shape_incoming:
        return layers.Normalization(input_shape=input_shape_incoming, axis=None)
    else:

        # return  tf.keras.layers.Normalization(axis=-1)
        return tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)

def plotResults(test_labels,predictions):
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.savefig('loss.png')

def gitReset(test_results):
    pervious_accuracy = None
    fileName = os.path.join(RESULTS_PATH,'Accuracy.txt')
    with open(fileName) as f:
        pervious_accuracy = float(f.readline())
    reset = False
    if test_results['dnn_model']<pervious_accuracy:
        subprocess.call(["git", "reset","--hard","HEAD~1"])
        print("REVERSED")
        return True
    else:
        return False
        

def writeResults(test_results):
    fileName = os.path.join(RESULTS_PATH,'Accuracy.txt')
    with open(fileName,'w') as outfile:
        outfile.write(str(test_results['dnn_model']))
