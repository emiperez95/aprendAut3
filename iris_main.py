
import numpy as np
from naive import Naive
from evaluation import Evaluation

CLASS_AMM = 3

trainingData = np.load("iris_data/trainingData.npy").astype(float)
competitionData = np.load("iris_data/competitionData.npy").astype(float)

model = Naive(trainingData, [1, 1, 1, 1])
eval = Evaluation(model, competitionData, CLASS_AMM)
eval.normalPrint()
