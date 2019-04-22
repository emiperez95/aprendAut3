
import numpy as np
from naive import Naive
from knn2 import Knn2
from evaluation import Evaluation
import time
import cProfile
import matplotlib.pyplot as plt
import scipy.stats as sp

CLASS_AMM_IRIS = 3
CLASS_AMM_COV = 7

# ====================== IRIS
trainingData = np.load("iris_data/trainingData.npy").astype(float)
trainData = trainingData
competitionData = np.load("iris_data/competitionData.npy").astype(float)

def KNNIris(Ks=[1,3,7]):
  testData1 = np.load("iris_data/competitionData.npy")
  testData2 = np.load("iris_data/evaluationData.npy")
  trainData = np.load("iris_data/trainingData.npy").astype(float)
  testData = np.concatenate((testData1, testData2)).astype(float)
  for k in Ks:
    model = Knn2(trainData, k)
    print('>>>> for K='+str(k)+':')
    ev = Evaluation(model, testData, CLASS_AMM_IRIS)
    ev.normalPrint()
    print()

def BayesIris():
  testData1 = np.load("iris_data/competitionData.npy")
  testData2 = np.load("iris_data/evaluationData.npy")
  trainData = np.load("iris_data/trainingData.npy").astype(float)
  testData = np.concatenate((testData1, testData2)).astype(float)

  model = Naive(trainData,[1,1,1,1])

  ev = Evaluation(model, testData, CLASS_AMM_IRIS)
  ev.normalPrint()
# ====================== COV_TYPE
trainingData500k = np.load('cov_data/500k/trainingData.npy')
competitionDataCov = np.load("cov_data/compDataCover.npy")
evaluationDataCov = np.load("cov_data/evaDataCover.npy")




# ======================== CORRELATION GRAPHICS ========================
# ======================== DO NOT DELETE THIS ==========================
def correlationCovType():
  classDict = {
    0: 'Elevation',
    1: 'Aspect',
    2: 'Slope',
    3: 'Horizontal_Distance_To_Hydrology',
    4: 'Vertical_Distance_To_Hydrology',
    5: 'Horizontal_Distance_To_Roadways',
    6: 'Hillshade_9am',
    7: 'Hillshade_Noon',
    8: 'Hillshade_3pm',
    9: 'Horizontal_Distance_To_Fire_Points',
  }

  it = 0
  for i in range(10):
    for j in range(i+1, 10):
      if it == 0 or it == 25:
        fig = plt.figure(i)
        fig.subplots_adjust(left=0.05, bottom=0.06, right=0.96, top=0.94, wspace=0.50, hspace=0.63)
        it = 0
        # Creo figura
      p = plt.subplot(5, 5, it+1)
      p.set_title('Corr: '+str(round(np.corrcoef(trainingData500k[:,i],trainingData500k[:,j])[0][1], 3)))
      p.set_xlabel(classDict[i], fontsize='xx-small')
      p.set_ylabel(classDict[j], fontsize='xx-small')
      corr = p.plot(trainingData500k[:15000,i],trainingData500k[:15000,j], 'ro')
      plt.setp(corr, markersize=0.5)
      it += 1
  plt.show()
# ======================== CORRELATION GRAPHICS ========================
# ======================== DO NOT DELETE THIS ==========================

# ======================== CORRELATION IRIS ========================
# ======================== DO NOT DELETE THIS ==========================
def correlationIris():
  classDict = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
  }

  trainingData500k = trainData

  it = 0
  for i in range(4):
    for j in range(i+1, 4):
      if it == 0 or it == 25:
        fig = plt.figure(i)
        fig.subplots_adjust(left=0.05, bottom=0.06, right=0.96, top=0.94, wspace=0.50, hspace=0.63)
        it = 0
        # Creo figura
      p = plt.subplot(3, 2, it+1)
      p.set_title('Corr: '+str(round(np.corrcoef(trainingData500k[:,i],trainingData500k[:,j])[0][1], 3)))
      p.set_xlabel(classDict[i], fontsize='xx-small')
      p.set_ylabel(classDict[j], fontsize='xx-small')
      corr = p.plot(trainingData500k[:,i],trainingData500k[:,j], 'ro')
      plt.setp(corr, markersize=0.5)
      it += 1
  plt.show()
# ======================== CORRELATION GRAPHICS ========================
# ======================== DO NOT DELETE THIS ==========================


# ======================================================================================
# ================================= K-NN EVALUATION ==============================
# ======================================================================================
def KNNCovType(knnSizes=[5, 50, 500], Ks=[1,3,7]):
  attTypes = [1 for _ in range(10)] + [0 for _ in range(44)]
  for size in knnSizes:
    print('\n\nLoading data from path='+"cov_data/"+str(size)+"k/trainingData.npy")
    trainingData = np.load("cov_data/"+str(size)+"k/trainingData.npy")
    print('- Dataset of ' + str(size) + 'k')
    for k in Ks:
      start = time.time()
      print('\n\n-- Train for k='+str(k))
      timer = time.time()
      model = Knn2(trainingData, k)
      evalModel = Evaluation(model, competitionDataCov, CLASS_AMM_COV)
      print('--- Evaluation for size='+str(size)+' and k='+str(k))
      evalModel.normalPrint()
      print('Took ', time.time() - start, 's')
# ====================================================================================
# ====================================================================================



# ======================================================================================
# ================================ NAIVE BAYES EVALUATION ==============================
# ======================================================================================
def NBCovType(NBSizes=[5, 50, 500]):
  attTypes = [1 for _ in range(10)] + [0 for _ in range(44)]
  for size in NBSizes:
    print('\n\nLoading data from path='+"cov_data/"+str(size)+"k/trainingData.npy")
    trainingData = np.load("cov_data/"+str(size)+"k/trainingData.npy")
    print('- Dataset of ' + str(size) + 'k')
    start = time.time()
    print('\n\n-- Train for NAIVE BAYES')
    model = Naive(trainingData, attTypes)
    timer = time.time()
    print('Time to train: ', timer - start)
    evalModel = Evaluation(model, competitionDataCov, CLASS_AMM_COV)
    print('--- Evaluation for size='+str(size))
    evalModel.normalPrint()
    print('Took ', time.time() - start, 's')
# ====================================================================================
# ====================================================================================


# ======================================================================================
# ================================ NAIVE BAYES CROSS VALIDATION ==============================
# ======================================================================================
def crossValidationCovType(k=5, size=500):
  attTypes = [1 for _ in range(10)] + [0 for _ in range(44)]
  wholeData = np.load("cov_data/"+str(size)+"k/trainingData.npy")
  folds = k
  trainingSets = np.split(wholeData, [(i+1)*(len(wholeData)//folds) for i in range(folds-1)])
  print([len(i) for i in trainingSets])

  for idx, testData in enumerate(trainingSets):
    print(np.r_[0:idx, idx+1:len(trainingSets)+1])
    trainingData = np.concatenate([trainingSets[i] for i in np.r_[0:idx, idx+1:len(trainingSets)]])
    print('K FOLD #'+str(idx))
    start = time.time()
    model = Naive(trainingData, attTypes)
    timer = time.time()
    print('Time to train: ', timer - start)
    start = time.time()
    evalModel = Evaluation(model, competitionDataCov, CLASS_AMM_COV)
    evalModel.normalPrint()
    print('Took ', time.time() - start, 's')


def crossValidationIris(k=5):
  testData1 = np.load("iris_data/competitionData.npy")
  testData2 = np.load("iris_data/evaluationData.npy")
  trainData = np.load("iris_data/trainingData.npy")
  wholeData = np.concatenate((trainData, testData1, testData2)).astype(float)
  folds = k
  attTypes = [1,1,1,1]
  trainingSets = np.split(wholeData, [(i+1)*round(len(wholeData)/folds) for i in range(folds-1)])

  for set in trainingSets:
    print('\n\n\n', set, '\n\n\n')

  for idx, testData in enumerate(trainingSets):
    print(np.r_[0:idx, idx+1:len(trainingSets)+1])
    trainingData = np.concatenate([trainingSets[i] for i in np.r_[0:idx, idx+1:len(trainingSets)]])
    print('testData: ', len(testData))
    print('trainingData: ', len(trainingData))
    print('K FOLD #'+str(idx))
    start = time.time()
    model = Naive(trainingData, attTypes)
    timer = time.time()
    print('Time to train: ', timer - start)
    start = time.time()
    evalModel = Evaluation(model, testData, CLASS_AMM_IRIS)
    evalModel.normalPrint()
    print('Took ', time.time() - start, 's')
# ======================================================================================
# ======================================================================================
# ======================================================================================


# ======================================================================================
# ============= DATA DISTRIBUTION   ====================================================
# ======================================================================================
def dataDistributionCovertype():
  classDict = {
    0: 'Elevation',
    1: 'Aspect',
    2: 'Slope',
    3: 'Horizontal_Distance_To_Hydrology',
    4: 'Vertical_Distance_To_Hydrology',
    5: 'Horizontal_Distance_To_Roadways',
    6: 'Hillshade_9am',
    7: 'Hillshade_Noon',
    8: 'Hillshade_3pm',
    9: 'Horizontal_Distance_To_Fire_Points',
  }

  wholeData = np.load("cov_data/500k/trainingData.npy")
  normalized = (wholeData[:,0] - np.mean(wholeData[:,0]))/np.ptp(wholeData[:,0])
  plt.subplot(4,3,1)
  plt.title(classDict[0])
  plt.hist(wholeData[:,0], color='blue', edgecolor='black', bins=100)
  plt.subplot(4,3,2)
  plt.title(classDict[1])
  plt.hist(wholeData[:,1], color='blue', edgecolor='black', bins=100)
  plt.subplot(4,3,3)
  plt.title(classDict[2])
  plt.hist(wholeData[:,2], color='blue', edgecolor='black', bins=100)
  plt.subplot(4,3,4)
  plt.title(classDict[3])
  plt.hist(wholeData[:,3], color='blue', edgecolor='black', bins=100)
  plt.subplot(4,3,5)
  plt.title(classDict[4])
  plt.hist(wholeData[:,4], color='blue', edgecolor='black', bins=100)
  plt.subplot(4,3,6)
  plt.title(classDict[5])
  plt.hist(wholeData[:,5], color='blue', edgecolor='black', bins=100)
  plt.subplot(4,3,7)
  plt.title(classDict[6])
  plt.hist(wholeData[:,6], color='blue', edgecolor='black', bins=100)
  plt.subplot(4,3,8)
  plt.title(classDict[7])
  plt.hist(wholeData[:,7], color='blue', edgecolor='black', bins=100)
  plt.subplot(4,3,9)
  plt.title(classDict[8])
  plt.hist(wholeData[:,8], color='blue', edgecolor='black', bins=100)
  plt.subplot(4,3,10)
  plt.title(classDict[9])
  plt.hist(wholeData[:,9], color='blue', edgecolor='black', bins=100)
  # plt.plot(sp.norm.pdf)
  plt.show()
# ======================================================================================
# ======================================================================================
# ======================================================================================


# ===============================================================================
# ====================== TIMES TO CLASSIFY AND TRAIN ============================
# ===============================================================================
# ======= Time to classify augmenting counts on Train Set =======
def takeMinTime(model, tupla):
  times = np.array([])
  for i in range(10):
    start = time.time()
    model.classify(tupla)
    ti = time.time() - start
    times = np.append(times, ti)
  return times.min()

def timeToClassify(count=360):
  attTypes = [1 for _ in range(10)] + [0 for _ in range(44)]
  print('===============================================================')
  print('======= Time to classify augmenting counts on Train Set =======')
  print('===============================================================')
  trainingData = np.load("cov_data/500k/trainingData.npy").astype(float)
  sizes = [i+1 for i in range(count)]
  knnTimes = []
  id3TrainTimes = []
  id3Times = []
  nbTrainTimes = []
  nbTimes = []
  tupla = competitionDataCov[2,:-1]
  for i in sizes:
    start = time.time()
    nb = Naive(trainingData[:i*1000], attTypes)
    ti = time.time() - start
    nbTrainTimes.append(ti)
    print('#',i,' NB Train - ', ti)
    minTime = takeMinTime(nb, tupla)
    nbTimes.append(minTime)
    print('#',i,' NB Classify - ', minTime)
    del nb

    start = time.time()
    id3 = ID3(trainingData[:i*1000], 54, True, 2, attTypes, 1)
    ti = time.time() - start
    id3TrainTimes.append(ti)
    print('#',i,' ID3 Train - ', ti)
    minTime = takeMinTime(id3, tupla)
    id3Times.append(minTime)
    print('#',i,' ID3 Classify - ', minTime)
    del id3

    knn = Knn2(trainingData[:i*1000], 7)
    minTime = takeMinTime(knn, tupla)
    knnTimes.append(minTime)
    print('#',i,' K-NN Classify - ', minTime)
    del knn

  plt.figure(1)
  plt.subplot(211)
  plt.ylabel('Time')
  plt.title('Naive Bayes train time')
  plt.plot(sizes, nbTrainTimes)

  plt.subplot(212)
  plt.ylabel('Time')
  plt.title('ID3 train time')
  plt.plot(sizes, id3TrainTimes)

  plt.figure(2)
  plt.subplot(311)
  plt.ylabel('Time')
  plt.title('Naive Bayes classify time')
  plt.plot(sizes, nbTimes)

  plt.subplot(312)
  plt.ylabel('Time')
  plt.title('ID3 classify time')
  plt.plot(sizes, id3Times)

  plt.subplot(313)
  plt.ylabel('Time')
  plt.title('K-NN classify time')
  plt.plot(sizes, knnTimes)

  plt.show()
# ===============================================================================
# ===============================================================================
opcion = '10'
while opcion != '0':
  print ("\n\nOpciones:")
  print ("1- Correr Bayes Ingenuo con conjunto Iris")
  print ("2- Correr Bayes Ingenuo con conjunto CoverType con 5K,50K y 500K")
  print ("3- Correr K-NN con conjunto Iris con k igual a 1,3 y 7")
  print ("4- Correr K-NN con conjunto CoverType con 5K, 50K y 500K con k igual a 1,3 y 7")
  print ("0- Salir")
  opcion = input("Qué opción deseas? >> ")
  print('\n')

  if opcion == '1':
    BayesIris()
  elif opcion == '2':
    NBCovType()
  elif opcion == '3':
    KNNIris()
  elif opcion == '4':
    KNNCovType()
  elif opcion != '0':
    print("Opción no válida")
