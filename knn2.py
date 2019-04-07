import numpy as np
from multiprocessing import Process, Value, Pool, Lock, Queue
import time

class Knn2:
  def __init__(self, data, attTypes, k, **kwargs):
    self.kwargs = kwargs
    self.data = data
    self.k = k

  def classify(self, tuple):
    k = self.k
    results = np.sqrt(np.sum((self.data[:,:-1]-tuple)**2, axis=1))
    return np.bincount(self.data[np.argsort(results)[:k]][:,-1].astype(int)).argmax()

  def distance(self, a, b):
    return self.euclidean_distance(a, b)

  def euclidean_distance(self, a, b):
    return np.sqrt(np.sum((a-b)**2))
