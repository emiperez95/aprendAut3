import numpy as np
import scipy.stats as sp
import pprint

class Naive:
    def __init__(self, data, attTypes, **kwargs): #Data tiene que ser de numpy
        self.kwargs = kwargs

        self.dataLen = len(data)
        self.attTypes = attTypes
        self.dataDist = {}
        self.classDist = {}
        classCol = data.T[-1]

        for clas in classCol:
            if clas not in self.classDist:
                self.classDist[clas] = 1
            else:
                self.classDist[clas] += 1

        for i, col in enumerate(data.T[:-1]):
            colType = attTypes[i]
            
            if colType == 0: #Valor cualitativo
                for j, cell in enumerate(col):
                    if cell not in self.dataDist[i]:
                        self.dataDist[i][cell] = {}
                    classJ = classCol[j]
                    if classJ not in self.dataDist[i][cell]:
                        self.dataDist[i][cell][classJ] = 1 
                    else:
                        self.dataDist[i][cell][classJ] += 1

                # for a in self.dataDist[i]:
                #     for key, val in self.classDist:#TODO:
                #         a[key] = a[key]/val
            else: # colType == 1: Valor cuantitativo
                classVal = {}
                for j, cell in enumerate(col):
                    classJ = classCol[j]
                    if classJ not in classVal:
                        classVal[classJ] = []
                    classVal[classJ].append(cell)
                for key, val in classVal:
                    self.dataDist[key] = {
                        "mean" : np.mean(val),
                        "stdv" : np.std(val)
                    }
                pass
        # for a in self.classDist: #TODO:
            # self.classDist[a] = self.classDist[a]/self.dataLength

    def classify(self, tupl):
        p = 1/self.dataLen
        m = 1
        probsSum = 0
        maxProb = 0
        maxargv = -1
        for clas, value in self.classDist:
            total = 1
            for i, elem in enumerate(tupl):
                total *= self.normal(self.dataDist[i], elem) if self.attTypes[i] == 1 else (self.dataDist[i] + m*p)/(value + m)
            probsSum += total
            if total > maxProb:
                maxProb = total
                maxargv = value
        return maxargv, maxProb/probsSum
        
    def normal(self, dic, value):
        return sp.stats.norm(dic['mean'], dic['stdv']).pdf(value) #TODO:



data = [
    [1, 2, 3, 1, 2, 3, 1],
    [2, 3, 5, 1, 3, 4, 0],
    [2, 3, 4, 5, 1, 2, 1],
    [2, 3, 5, 1, 3, 4, 1],
    [2, 3, 5, 1, 3, 4, 1],
    [2, 3, 5, 1, 3, 4, 0],
    [2, 3, 5, 1, 3, 4, 0],
    [3, 4, 6, 2, 5, 6, 1]
]
attTypes = [0, 0, 0, 0, 0, 0] 

a = Naive(np.array(data), attTypes)
pprint(a.dataDist)
pprint(a.classDist)