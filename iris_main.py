
import numpy as np
from naive import Naive
from evaluation import Evaluation
import time
import cProfile

CLASS_AMM_IRIS = 3
CLASS_AMM_COV = 7

# trainingData = np.load("iris_data/trainingData.npy").astype(float)
# competitionData = np.load("iris_data/competitionData.npy").astype(float)

# model = Naive(trainingData, [1, 1, 1, 1])
# eval = Evaluation(model, competitionData, CLASS_AMM_IRIS)
# eval.normalPrint()

# ====================== COV_TYPE
competitionDataCov = (np.load("cov_data/compDataCover.npy"))
attTypes = [1 for _ in range(10)] + [1 for _ in range(44)]

# #-------------5k-------------
# print("===================5k======================")
# timer5k = time.time()
# trainingData5k = np.load("cov_data/5k/trainingData.npy")
# cov5kModel = Naive(trainingData5k, attTypes)
# evalTimer5k = time.time()
# print("TrainTime: ", evalTimer5k-timer5k)
# eval5k = Evaluation(cov5kModel, competitionDataCov, CLASS_AMM_COV)
# eval5k.normalPrint()
# print("ExecutionTime: ", time.time()-evalTimer5k)

# -------------50k-------------
# print("===================50k======================")
# timer50k = time.time()
# trainingData50k = np.load("cov_data/50k/trainingData.npy")
# cov50kModel = Naive(trainingData50k, attTypes)
# evalTimer50k = time.time()
# print("TrainTime: ", evalTimer50k-timer50k)
# eval50k = Evaluation(cov50kModel, competitionDataCov, CLASS_AMM_COV)
# eval50k.normalPrint()
# print("ExecutionTime: ", time.time()-evalTimer50k)

#-------------500k-------------
print("===================500k======================")
timer500k = time.time()
trainingData500k = np.load("cov_data/500k/trainingData.npy")
cov500kModel = Naive(trainingData500k, attTypes)
evalTimer500k = time.time()
print("TrainTime: ", evalTimer500k-timer500k)
eval500k = Evaluation(cov500kModel, competitionDataCov, CLASS_AMM_COV)
eval500k.normalPrint()
print("ExecutionTime: ", time.time()-evalTimer500k)

# print("=~=~=~=~=~=~=~=~=~=~=AttType~=~=~=~=~=~=~=~=~=~=~=~=")
# # ====================== COV_TYPE
# competitionDataCov = np.load("cov_data/compDataCover.npy")
# attTypes = [1 for _ in range(10)] + [1 for _ in range(44)]

# #-------------5k-------------
# print("===================5k======================")
# timer5k = time.time()
# trainingData5k = np.load("cov_data/5k/trainingData.npy")
# cov5kModel = Naive(trainingData5k, attTypes)
# evalTimer5k = time.time()
# print("TrainTime: ", evalTimer5k-timer5k)
# eval5k = Evaluation(cov5kModel, competitionDataCov, CLASS_AMM_COV)
# eval5k.normalPrint()
# print("ExecutionTime: ", time.time()-evalTimer5k)

# #-------------50k-------------
# print("===================50k======================")
# timer50k = time.time()
# trainingData50k = np.load("cov_data/50k/trainingData.npy")
# cov50kModel = Naive(trainingData50k, attTypes)
# evalTimer50k = time.time()
# print("TrainTime: ", evalTimer50k-timer50k)
# eval50k = Evaluation(cov50kModel, competitionDataCov, CLASS_AMM_COV)
# eval50k.normalPrint()
# print("ExecutionTime: ", time.time()-evalTimer50k)

# #-------------500k-------------
# print("===================500k======================")
# timer500k = time.time()
# trainingData500k = np.load("cov_data/500k/trainingData.npy")
# cov500kModel = Naive(trainingData500k, attTypes)
# evalTimer500k = time.time()
# print("TrainTime: ", evalTimer500k-timer500k)
# eval500k = Evaluation(cov500kModel, competitionDataCov, CLASS_AMM_COV)
# eval500k.normalPrint()
# print("ExecutionTime: ", time.time()-evalTimer500k)