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
        self.__normPdf__und = np.sqrt(2*np.pi)
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
                        "std" : np.std(val)
                    }

    # @profile
    def classify(self, tupl):
        p = 1/self.dataLen
        m = 1
        # probsSum = 0
        maxProb = None
        maxargv = -1
        for j, value in self.classDist.items():
            total = np.log(1)
            for i, elem in enumerate(tupl):
                if self.attTypes[i] == 0:
                    if elem not in self.dataDist[i] or j not in self.dataDist[i][elem]:
                        cellCount = 0
                    else:
                        cellCount = self.dataDist[i][elem][j]
                    multiplier = np.log((cellCount + m*p))-np.log((value + m))
                else:
                    multiplier = self.normal(self.dataDist[i][j], elem)
                if multiplier == None:
                    multiplier = np.log(m*p)-np.log((value + m))
                total += multiplier
                # print("Total: {}, var {}, att{}".format(multiplier, j, i))
            # probsSum += np.power(np.e,total)
            if maxProb == None or total > maxProb:
                maxProb = total
                maxargv = j
        return maxargv#, np.power(np.e,maxProb)/probsSum

    # @profile
    def normal(self, dic, value):
        if dic["std"] == 0:
            if value == dic["mean"]:
                return 0
            else:
                return None
        else:
            # norm = dic["norm"]
            # retVal = norm.logpdf(value)

            std = dic["std"]
            mean = dic['mean']
            retVal = self.__logVal(std, mean, value)

            # if abs(a - retVal) > 0.01 or True:
            #     print(a, retVal)
            #     input("")
            return retVal

    # @profile
    def __logVal(self, std, mean, val):
        und = np.log(std * self.__normPdf__und)
        exp = ((val - mean)**2)/(2*(std**2))
        return - exp - und

    def showDists(self):
        print('>> Attributes distributions:')
        for i in self.dataDist:
            print('- Attribute ', i)
            for cl in self.dataDist:
                print('-- Class ', cl)
                print('--- Mean: ', self.dataDist[i][cl]['mean'])
                print('--- Std: ', self.dataDist[i][cl]['std'])

# data = [
#     [1, 2, 3, 1, 2, 3, 1],
#     [2, 3, 5, 1, 3, 4, 0],
#     [2, 3, 4, 5, 1, 2, 1],
#     [2, 3, 5, 1, 3, 4, 1],
#     [2, 3, 5, 1, 3, 4, 1],
#     [2, 3, 5, 1, 3, 4, 0],
#     [2, 3, 5, 1, 3, 4, 0],
#     [3, 4, 6, 2, 5, 6, 1]
# ]
# data = [
#     [0, 0, 1, 2, 3, 0],
#     [0, 1, 2, 2, 3, 1],
#     [1, 0, 1, 2, 3, 0],
#     [1, 1, 2, 2, 3, 1]
# ]
# attTypes = [1, 1, 0, 0, 0]

# a = Naive(np.array(data), attTypes)
# pp.pprint(a.dataDist)
# pp.pprint(a.classDist)
# print(a.classify([0.5, 0, 2, 2, 3]))
