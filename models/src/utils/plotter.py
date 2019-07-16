import matplotlib.pyplot as plt
from seaborn import heatmap
import numpy as np

def plot_loss_graph(training_loss, test_loss):
    plt.plot(training_loss, label = "Training")
    plt.plot(test_loss, label = "Test")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.show()

def plot_confusion_matrix(matrix):
    labels = [0,1,2]
    matrix = np.int32(matrix)
    fig, ax = plt.subplots(figsize=(15, 10))
    heatmap(matrix, annot=True, fmt = "d",
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()