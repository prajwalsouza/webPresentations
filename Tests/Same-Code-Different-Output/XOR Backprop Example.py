import random 
import math
from matplotlib import pyplot as plt

weightvalue = {}
weightUpdate = {}

# First entry is the bias neuron defaulted to 1

inputlayer = [1,1,1]
hiddenlayer = [1,1,1,1,1,1]
outputlayer = [1]


for k in range(1,len(hiddenlayer)):
	for i in range(len(inputlayer)):
		weightvalue[str(k) + '-' + str(i) + '-' + str(1)] = 1
		weightUpdate[str(k) + '-' + str(i) + '-' + str(1)] = 0

for k in range(len(outputlayer)):
	for i in range(len(hiddenlayer)):
		weightvalue[str(k) + '-' + str(i) + '-' + str(2)] = 1
		weightUpdate[str(k) + '-' + str(i) + '-' + str(2)] = 0


# Sigmoid Definition
def sigmoid(x):
	return 1/(1 + math.exp(-x))


def feedForward(x1,x2):
	inputlayer[1] = x1
	inputlayer[2] = x2

	for k in range(1,len(hiddenlayer)):
		sumvalue = 0
		for i in range(len(inputlayer)):
			sumvalue = sumvalue + weightvalue[str(k) + '-' + str(i)+ '-1']*inputlayer[i]
		hiddenlayer[k] = sigmoid(sumvalue)

	
	for k in range(len(outputlayer)):
		sumvalue = 0
		for i in range(len(hiddenlayer)):
			sumvalue = sumvalue + weightvalue[str(k) + '-' + str(i)+ '-2']*hiddenlayer[i]
		outputlayer[k] = sigmoid(sumvalue)



	return outputlayer[0]


def ANDfunction(x1,x2):
	x1 = int(x1)
	x2 = int(x2)

	if(x1 == 0 and x2 == 0):
		return 0
	elif(x1 == 1 and x2 == 0):
		return 0
	elif(x1 == 0 and x2 == 1):
		return 0
	elif(x1 == 1 and x2 == 1):
		return 1
	else:
		return 'Error in Inputs.'

def XORfunction(x1,x2):
	x1 = int(x1)
	x2 = int(x2)

	if(x1 == 0 and x2 == 0):
		return 0
	elif(x1 == 1 and x2 == 0):
		return 1
	elif(x1 == 0 and x2 == 1):
		return 1
	elif(x1 == 1 and x2 == 1):
		return 0
	else:
		return 'Error in Inputs.'


def errorValue(actual, ideal):
	return (actual - ideal)


def sigmoidDerivative(sigmoidValue):
	return sigmoidValue*(1 - sigmoidValue)


def nodeDelta(errorvalue, derivativeValue, neurontype, deltaslist, weightslist):
	if neurontype == 'outputNeuron':
		return -errorvalue*derivativeValue
	if neurontype == 'interiorNeuron':
		sumvalue = 0
		for i in range(len(deltaslist)):
			sumvalue = sumvalue + deltaslist[i]*weightslist[i]
		return sumvalue*derivativeValue

errorGradientOfWeight = {}


def backProp(itrs):
	feedForward(trainingpairs[itrs%4][0],trainingpairs[itrs%4][1])
	xorValue = ANDfunction(inputlayer[1],inputlayer[2])
	error = errorValue(outputlayer[0],xorValue)

	nodeDeltaHiddenLayer = []
	for i in range(len(hiddenlayer)):
		nodeDeltaHiddenLayer.append(0)

	nodeDeltaOutputLayer = []
	for i in range(len(outputlayer)):
		nodeDeltaOutputLayer.append(0)

	derivative = sigmoidDerivative(outputlayer[0])

	for i in range(len(outputlayer)):
		nodeDeltaOutputLayer[i] = nodeDelta(error, derivative, 'outputNeuron',[],[])

	for i in range(1,len(hiddenlayer)):
		deltas = []
		weights = []
		for k in range(len(outputlayer)):
			deltas.append(nodeDeltaOutputLayer[k])
			weights.append(weightvalue[str(k) + '-' + str(i) + '-2'])
		derivative = sigmoidDerivative(hiddenlayer[i])
		nodeDeltaHiddenLayer[i] = nodeDelta(error, derivative, 'interiorNeuron',deltas, weights)



	for k in range(1,len(hiddenlayer)):
		for i in range(len(inputlayer)):
			errorGradientOfWeight[str(k) + '-' + str(i) + '-' + str(1)] = nodeDeltaHiddenLayer[k]*inputlayer[i]

	for k in range(len(outputlayer)):
		for i in range(len(hiddenlayer)):
			errorGradientOfWeight[str(k) + '-' + str(i) + '-' + str(2)] = nodeDeltaOutputLayer[k]*hiddenlayer[i]

	for k in range(1,len(hiddenlayer)):
		for i in range(len(inputlayer)):
			weightUpdate[str(k) + '-' + str(i) + '-' + str(1)] = (epsilon*errorGradientOfWeight[str(k) + '-' + str(i) + '-' + str(1)]) + (alpha*weightUpdate[str(k) + '-' + str(i) + '-' + str(1)])
			weightvalue[str(k) + '-' + str(i) + '-' + str(1)] = weightUpdate[str(k) + '-' + str(i) + '-' + str(1)] + weightvalue[str(k) + '-' + str(i) + '-' + str(1)]
			print('{0:.23f}'.format(weightUpdate[str(k) + '-' + str(i) + '-' + str(1)]))

	for k in range(len(outputlayer)):
		for i in range(len(hiddenlayer)):
			weightUpdate[str(k) + '-' + str(i) + '-' + str(2)] = (epsilon*errorGradientOfWeight[str(k) + '-' + str(i) + '-' + str(2)]) + (alpha*weightUpdate[str(k) + '-' + str(i) + '-' + str(2)])
			weightvalue[str(k) + '-' + str(i) + '-' + str(2)] = weightUpdate[str(k) + '-' + str(i) + '-' + str(2)] + weightvalue[str(k) + '-' + str(i) + '-' + str(2)]

	# print(weightvalue)
	return error

epsilon = 0.7
alpha = 0.3

errors = []
trainingpairs = [[0,0],[0,1],[1,0],[1,1]]

for iterC in range(50):
	print(iterC)
	er = backProp(iterC)
	errors.append(abs(er))
	print('{0:.23f}'.format(abs(er)))


for i in range(4):
	x1value = trainingpairs[i%4][0]
	x2value = trainingpairs[i%4][1]
	actualValue = feedForward(x1value,x2value)

	idealValue = ANDfunction(x1value,x2value)
	print("The Output of the Feed Foward Propagation with inputs \n%d to input Neuron X1 and %d to input Neuron X2 is %f. Ideal Value is %d." % (x1value,x2value,actualValue,idealValue))


# plt.plot(errors)
# plt.show()