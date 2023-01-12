import numpy as np
class NN:
    def __init__(self, structure):
        self.structure = structure
        
        self.values = [] #[layer][number]
        self.biases = [] #[layer][number]
        self.weights = [] #[layer][to][from]

        ##
        for i in range(len(self.structure)):
            tempv = []
            for j in range(self.structure[i]):
                tempv.append(0)
            self.values.append(tempv)
            self.biases.append(tempv)

        ##
        for i in range(len(self.structure) - 1):
            temp2v = [] #2d vector
            for j in range(self.structure[i+1]):
                tempv = []
                for k in range(structure[i]):
                    tempv.append(np.random.rand(1) * np.sqrt(2 / self.structure[i]))
                temp2v.append(tempv)
            self.weights.append(temp2v)


    def Test(self, inputs):
        for i in range(len(self.values[0])): self.values[0][i] = inputs[i]

        #iterate over every node
        for i in range(1, len(self.values)):
            for j in range(len(self.values[i])):
                #set their values
                self.values[i][j] = self.HardSigmoid( self.Sum( self.values[i - 1], self.weights[i - 1][j]) + self.biases[i][j])

        return self.values[len(self.values) - 1]

    
    #def SigmoidDerivative(self, x):

    def Sum(self, values, weights):
        res = 0
        for i in range(len(values)):
            res += values[i] * weights[i]

        return res

    def HardSigmoid(self, x): return x / (1 + np.abs(x))
