import numpy as np
from multiprocessing import Process, Value, Pool, Lock, Queue
import time


class Knn2:
  def __init__(self, data, attTypes, **kwargs):
    self.kwargs = kwargs
    self.data = data

  # def classify(self, tuple, k):
  #   results = []
  #   positives = 0
  #   for ind, col in enumerate(self.data[:,:-1]):
  #     results.append([self.distance(tuple[:-1], col), data[ind][-1]])
  #   results = np.array(results)
  #   results = results[np.argsort(results[:,0])]
  #   firstK = results[:,-1][:k].astype(int)
  #   return np.bincount(firstK).argmax()

  def classify(self, tuple, k):
    results = np.sqrt(np.sum((self.data-tuple)**2, axis=1))
    return np.bincount(self.data[np.argsort(results)[:k]][:,-1].astype(int)).argmax()

  def distance(self, a, b):
    return self.euclidean_distance(a, b)

  def euclidean_distance(self, a, b):
    return np.sqrt(np.sum((a-b)**2))

attTypes = [1 for i in range(54)]

DATA_LOCATION = "cov_data"
evaluationData = DATA_LOCATION + "/evaDataCover.npy"
trainingData = DATA_LOCATION + "/50k/trainingData.npy"
competitionData = DATA_LOCATION + "/competitionData.npy"

data = np.load(trainingData).astype(float)
evData = np.load(evaluationData).astype(float)

knn = Knn2(data, attTypes)
results = {
  'pos': 0,
  'neg': 0,
}


def iter(size, time, q):
  fromm = (size-1)*time
  to = fromm + size
  pos = 0
  neg = 0
  for ind, row in enumerate(evData[fromm:to]):
    predicted = knn.classify(row, 5)
    if predicted == int(row[-1]):
      pos += 1
    else:
      neg += 1
    print('#',ind + ind*time,'Tuple class: ', row[-1], '- Predicted as ', predicted)
  q.put([pos, neg])

start = time.time()
pos = {}
neg = {}
q = Queue()
p = []
sizeOfIter = 0
times = 5
lock = Lock()
for i in range(times):
  pos = Value('i', 0)
  neg = Value('i', 0)
  p.append(Process(target=iter, args=(sizeOfIter, i, q)))
  p[i].start()

pos = 0
neg = 0
for i in range(times):
  p[i].join()

for i in range(times):
  print(q.get())
end = time.time()
time1 = end - start
print('With mutliprocessing Took ', time1)
# start = time.time()
# pos = Value('i', 0)
# neg = Value('i', 0)
# p = Process(target=iter, args=(10, 1, pos, neg))
# p.start()
# p.join()
# print(pos.value)
# print(neg.value)


start = time.time()
evaluationData = evData[:10000]
for ind, tuple in enumerate(evaluationData):
  predicted = knn.classify(tuple, 5)
  if predicted == int(tuple[-1]):
    results['pos'] += 1
  else:
    results['neg'] += 1
  print('#',ind,'Tuple class: ', tuple[-1], '- Predicted as ', predicted, '(', (results['pos']), '/',  ind+1, ')')
end = time.time()
print('With mutliprocessing Took ', time1)
print('Without mutliprocessing Took ', end - start)

# print(' == RESULTS == ')
# print('Correct: ', results['pos'], '/', len(evaluationData))
# print('Percentage: ', results['pos']*100/len(evaluationData))
# print(Knn2(data, attTypes).classify(evaluationData[0], 3))
# print('Took ', end - start, ' seconds')
