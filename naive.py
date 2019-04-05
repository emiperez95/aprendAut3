import numpy as np
import scipy.stats as sp
import pprint as pp

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
            self.dataDist[i] = {}
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
                for key, val in classVal.items():
                    self.dataDist[i][key] = {
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
        maxProb = None
        maxargv = -1
        for j, value in self.classDist.items():
            total = np.log(1)
            for i, elem in enumerate(tupl):
                if self.attTypes[i] == 0:
                    # print(i, elem, j)
                    if j in self.dataDist[i][elem]:
                        cellCount = self.dataDist[i][elem][j]
                    else:
                        cellCount = 0
                multiplier = self.normal(self.dataDist[i][j], elem) if self.attTypes[i] == 1 else np.log((cellCount + m*p))-np.log((value + m))
                if multiplier == None:
                    multiplier = np.log(m*p)-np.log((value + m))
                total += multiplier
                # print("Total: {}, var {}, att{}".format(multiplier, j, i))
            probsSum += np.power(np.e,total)
            print(j, total)
            if maxProb == None or total > maxProb:
                maxProb = total
                maxargv = j
        return maxargv, np.power(np.e,maxProb)/probsSum
        
    def normal(self, dic, value):
        if dic["stdv"] == 0:
            if value == dic["mean"]:
                return 0
            else:
                return None
        else:
            return sp.norm(dic["mean"], dic["stdv"]).logpdf(value)



# # data = [
# #     [1, 2, 3, 1, 2, 3, 1],
# #     [2, 3, 5, 1, 3, 4, 0],
# #     [2, 3, 4, 5, 1, 2, 1],
# #     [2, 3, 5, 1, 3, 4, 1],
# #     [2, 3, 5, 1, 3, 4, 1],
# #     [2, 3, 5, 1, 3, 4, 0],
# #     [2, 3, 5, 1, 3, 4, 0],
# #     [3, 4, 6, 2, 5, 6, 1]
# # ]
# data = [
#     [0, 0, 1, 2, 3, 0],
#     [0, 1, 1, 2, 3, 1],
#     [1, 0, 1, 2, 3, 0],
#     [1, 1, 1, 2, 3, 1]
# ]
# attTypes = [1, 1, 0, 0, 0] 

# a = Naive(np.array(data), attTypes)
# pp.pprint(a.dataDist)
# pp.pprint(a.classDist)
# print(a.classify([0.5, 0.5, 1, 2, 3]))
