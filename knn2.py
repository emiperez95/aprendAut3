import numpy as np
from multiprocessing import Process, Value, Pool, Lock, Queue
import time

class Knn2:
  def __init__(self, data, k, **kwargs):
    self.kwargs = kwargs
    self.data = data
    self.k = k

  def classify(self, tupla):
    # if tupla in self.data[:,:-1].tolist():
    #   print('ALERT')
    # else:
    #   print('NOT IN TRAIN SET')
    k = self.k
    results = np.sqrt(np.sum((self.data[:,:-1]-tupla)**2, axis=1))
    # print('PREDICTING TUPLE\n', tupla)
    # print(self.data[np.argsort(results)[:k]])
    # input()
    return np.bincount(self.data[np.argsort(results)[:k]][:,-1].astype(int)).argmax()

  def distance(self, a, b):
    return self.euclidean_distance(a, b)

  def euclidean_distance(self, a, b):
    return np.sqrt(np.sum((a-b)**2))
