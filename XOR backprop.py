# A Neural Network that mimmicks the AND function

""" The network has an input layer with two neurons, a hidden layer with two neuron and an output layer with a single neuron """

import random
import math
from matplotlib import pyplot as plt

# Initializing Neurons

inputNeuronX1 = 1
inputNeuronX2 = 1

firstLayerNeuronA1 = 1
firstLayerNeuronA2 = 1

outputNeuronY1 = 1

# Initializing Weights

weightToLayerOneNeuronA1FromInputNeuronX1 = random.uniform(-0.9,0.9)
weightToLayerOneNeuronA1FromInputNeuronX2 = random.uniform(-0.9,0.9)

firstLayerNeuronA1Bias = random.uniform(-0.9,0.9)

weightToLayerOneNeuronA2FromInputNeuronX1 = random.uniform(-0.9,0.9)
weightToLayerOneNeuronA2FromInputNeuronX2 = random.uniform(-0.9,0.9)

firstLayerNeuronA2Bias = random.uniform(-0.9,0.9)


weightToOutputNeuronY1FromLayerOneNeuronA1 = random.uniform(-0.9,0.9)
weightToOutputNeuronY1FromLayerOneNeuronA2 = random.uniform(-0.9,0.9)

outputLayerNeuronY1Bias = random.uniform(-0.9,0.9)


# Sigmoid Definition

def sigmoid(x):
	return 1/(1 + math.exp(-x))

# Forward Propagation

def feedForwardPropagation(inputX1, inputX2):
	global inputNeuronX1, inputNeuronX2, firstLayerNeuronA1Activation, firstLayerNeuronA2Activation, outputNeuronY1Activation, firstLayerNeuronA1Bias, firstLayerNeuronA2Bias, outputLayerNeuronY1Bias, weightToOutputNeuronY1FromLayerOneNeuronA1, weightToOutputNeuronY1FromLayerOneNeuronA2, weightToLayerOneNeuronA1FromInputNeuronX1, weightToLayerOneNeuronA1FromInputNeuronX2, weightToLayerOneNeuronA2FromInputNeuronX1, weightToLayerOneNeuronA2FromInputNeuronX2

	inputNeuronX1 = inputX1
	inputNeuronX2 = inputX2

	weightProductsToLayerOneNeuronA1 = (weightToLayerOneNeuronA1FromInputNeuronX1*inputNeuronX1) + (weightToLayerOneNeuronA1FromInputNeuronX2*inputNeuronX2)
	linearInputToLayerOneNeuronA1 = weightProductsToLayerOneNeuronA1 + firstLayerNeuronA1Bias
	firstLayerNeuronA1Activation = sigmoid(linearInputToLayerOneNeuronA1)

	weightProductsToLayerOneNeuronA2 = (weightToLayerOneNeuronA2FromInputNeuronX1*inputNeuronX1) + (weightToLayerOneNeuronA2FromInputNeuronX2*inputNeuronX2)
	linearInputToLayerOneNeuronA2 = weightProductsToLayerOneNeuronA2 + firstLayerNeuronA2Bias
	firstLayerNeuronA2Activation = sigmoid(linearInputToLayerOneNeuronA2)


	weightProductsToOutputNeuronY1 = (weightToOutputNeuronY1FromLayerOneNeuronA1*firstLayerNeuronA1Activation) + (weightToOutputNeuronY1FromLayerOneNeuronA2*firstLayerNeuronA2Activation)
	linearInputToOutputNeuronY1 = weightProductsToOutputNeuronY1 + outputLayerNeuronY1Bias
	outputNeuronY1Activation = sigmoid(linearInputToOutputNeuronY1)

	return outputNeuronY1Activation

	# print("The Output of the Feed Foward Propagation with inputs \n%f to input Neuron X1 and %f to input Neuron X2 is %f." % (inputNeuronX1,inputNeuronX2,outputNeuronY1Activation))



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


# for i in range(100):
# 	x1Value = random.randint(0,1)
# 	x2Value = random.randint(0,1)
# 	actualValue = feedForwardPropagation(x1Value,x2Value)

# 	idealValue = ANDfunction(x1Value,x2Value)

# 	error = errorValue(actualValue, idealValue)
	

# Back Propagation Algorithm

def nodeDelta(errorvalue, derivativeValue, neurontype, deltas, weights):
	if neurontype == 'outputNeuron':
		return -errorvalue*derivativeValue
	if neurontype == 'interiorNeuron':
		sumvalue = 0
		for i in range(len(deltas)):
			sumvalue = sumvalue + deltas[i]*weights[i]
		return sumvalue*derivativeValue

def sigmoidDerivative(sigmoidValue):
	return sigmoidValue*(1 - sigmoidValue)

weightToLayerOneNeuronA1FromInputNeuronX1Update = 0
weightToLayerOneNeuronA1FromInputNeuronX2Update = 0
weightToLayerOneNeuronA2FromInputNeuronX1Update = 0
weightToLayerOneNeuronA2FromInputNeuronX2Update = 0
weightToOutputNeuronY1FromLayerOneNeuronA1Update = 0
weightToOutputNeuronY1FromLayerOneNeuronA2Update = 0
firstLayerNeuronA1BiasUpdate = 0
firstLayerNeuronA2BiasUpdate = 0
outputLayerNeuronY1BiasUpdate = 0

def backPropagation(traintype):
	global inputNeuronX1, inputNeuronX2, firstLayerNeuronA1Activation, firstLayerNeuronA2Activation, outputNeuronY1Activation, firstLayerNeuronA1Bias, firstLayerNeuronA2Bias, outputLayerNeuronY1Bias, weightToOutputNeuronY1FromLayerOneNeuronA1, weightToOutputNeuronY1FromLayerOneNeuronA2, weightToLayerOneNeuronA1FromInputNeuronX1, weightToLayerOneNeuronA1FromInputNeuronX2, weightToLayerOneNeuronA2FromInputNeuronX1, weightToLayerOneNeuronA2FromInputNeuronX2
	global weightToLayerOneNeuronA1FromInputNeuronX1Update,weightToLayerOneNeuronA1FromInputNeuronX2Update,weightToLayerOneNeuronA2FromInputNeuronX1Update,weightToLayerOneNeuronA2FromInputNeuronX2Update,weightToOutputNeuronY1FromLayerOneNeuronA1Update,weightToOutputNeuronY1FromLayerOneNeuronA2Update,firstLayerNeuronA1BiasUpdate,firstLayerNeuronA2BiasUpdate,outputLayerNeuronY1BiasUpdate
	if (traintype == 'online'):
		nodeDeltaLayerOneNeuronA1 = 0
		nodeDeltaLayerOneNeuronA2 = 0
		nodeDeltaOutputNeuronY1 = 0

		# Forward Prop

		x1Value = random.randint(0,1)
		x2Value = random.randint(0,1)
		actualValue = feedForwardPropagation(x1Value,x2Value)

		idealValue = XORfunction(x1Value,x2Value)

		error = errorValue(actualValue, idealValue)
		errors.append(abs(error))

		derivative = sigmoidDerivative(outputNeuronY1Activation)

		nodeDeltaOutputNeuronY1 = nodeDelta(error, derivative, 'outputNeuron',[],[])


		weightsFromLayerOneNeuronA1 = [weightToOutputNeuronY1FromLayerOneNeuronA1]
		deltasFromLayerOneNeuronA1 = [nodeDeltaOutputNeuronY1]
		derivative = sigmoidDerivative(firstLayerNeuronA1Activation)
		nodeDeltaLayerOneNeuronA1 = nodeDelta(error, derivative, 'interiorNeuron', deltasFromLayerOneNeuronA1, weightsFromLayerOneNeuronA1)

		weightsFromLayerOneNeuronA2 = [weightToOutputNeuronY1FromLayerOneNeuronA2]
		deltasFromLayerOneNeuronA2 = [nodeDeltaOutputNeuronY1]
		derivative = sigmoidDerivative(firstLayerNeuronA2Activation)
		nodeDeltaLayerOneNeuronA2 = nodeDelta(error, derivative, 'interiorNeuron', deltasFromLayerOneNeuronA2, weightsFromLayerOneNeuronA2)

		# Gradients

		errorGradientWRTweightToOutputNeuronY1FromLayerOneNeuronA1 = firstLayerNeuronA1Activation*nodeDeltaOutputNeuronY1
		errorGradientWRTweightToOutputNeuronY1FromLayerOneNeuronA2 = firstLayerNeuronA2Activation*nodeDeltaOutputNeuronY1

		errorGradientWRTweightToLayerOneNeuronA1FromInputNeuronX1 = inputNeuronX1*nodeDeltaLayerOneNeuronA1
		errorGradientWRTweightToLayerOneNeuronA1FromInputNeuronX2 = inputNeuronX2*nodeDeltaLayerOneNeuronA1

		errorGradientWRTweightToLayerOneNeuronA2FromInputNeuronX1 = inputNeuronX1*nodeDeltaLayerOneNeuronA2
		errorGradientWRTweightToLayerOneNeuronA2FromInputNeuronX2 = inputNeuronX2*nodeDeltaLayerOneNeuronA2

		errorGradientWRTfirstLayerNeuronA1Bias = firstLayerNeuronA1Bias*nodeDeltaLayerOneNeuronA1
		errorGradientWRTfirstLayerNeuronA2Bias = firstLayerNeuronA2Bias*nodeDeltaLayerOneNeuronA2

		errorGradientWRToutputLayerNeuronY1Bias = outputLayerNeuronY1Bias*nodeDeltaOutputNeuronY1

		

		weightToLayerOneNeuronA1FromInputNeuronX1Update = (epsilon*errorGradientWRTweightToLayerOneNeuronA1FromInputNeuronX1) + (alpha*weightToLayerOneNeuronA1FromInputNeuronX1Update)
		weightToLayerOneNeuronA1FromInputNeuronX2Update = (epsilon*errorGradientWRTweightToLayerOneNeuronA1FromInputNeuronX2) + (alpha*weightToLayerOneNeuronA1FromInputNeuronX2Update)
		weightToLayerOneNeuronA2FromInputNeuronX1Update = (epsilon*errorGradientWRTweightToLayerOneNeuronA2FromInputNeuronX1) + (alpha*weightToLayerOneNeuronA2FromInputNeuronX1Update)
		weightToLayerOneNeuronA2FromInputNeuronX2Update = (epsilon*errorGradientWRTweightToLayerOneNeuronA2FromInputNeuronX2) + (alpha*weightToLayerOneNeuronA2FromInputNeuronX2Update)
		weightToOutputNeuronY1FromLayerOneNeuronA1Update = (epsilon*errorGradientWRTweightToOutputNeuronY1FromLayerOneNeuronA1) + (alpha*weightToOutputNeuronY1FromLayerOneNeuronA1Update)
		weightToOutputNeuronY1FromLayerOneNeuronA2Update = (epsilon*errorGradientWRTweightToOutputNeuronY1FromLayerOneNeuronA2) + (alpha*weightToOutputNeuronY1FromLayerOneNeuronA2Update)
		firstLayerNeuronA1BiasUpdate = (epsilon*errorGradientWRTfirstLayerNeuronA1Bias) + (alpha*firstLayerNeuronA1BiasUpdate)
		firstLayerNeuronA2BiasUpdate = (epsilon*errorGradientWRTfirstLayerNeuronA2Bias) + (alpha*firstLayerNeuronA2BiasUpdate)
		outputLayerNeuronY1BiasUpdate = (epsilon*errorGradientWRToutputLayerNeuronY1Bias) + (alpha*outputLayerNeuronY1BiasUpdate)
		
		weightToLayerOneNeuronA1FromInputNeuronX1 = weightToLayerOneNeuronA1FromInputNeuronX1 + weightToLayerOneNeuronA1FromInputNeuronX1Update
		weightToLayerOneNeuronA1FromInputNeuronX2 = weightToLayerOneNeuronA1FromInputNeuronX2 + weightToLayerOneNeuronA1FromInputNeuronX2Update
		weightToLayerOneNeuronA2FromInputNeuronX1 = weightToLayerOneNeuronA2FromInputNeuronX1 + weightToLayerOneNeuronA2FromInputNeuronX1Update
		weightToLayerOneNeuronA2FromInputNeuronX2 = weightToLayerOneNeuronA2FromInputNeuronX2 + weightToLayerOneNeuronA2FromInputNeuronX2Update
		weightToOutputNeuronY1FromLayerOneNeuronA1 = weightToOutputNeuronY1FromLayerOneNeuronA1 + weightToOutputNeuronY1FromLayerOneNeuronA1Update
		weightToOutputNeuronY1FromLayerOneNeuronA2 = weightToOutputNeuronY1FromLayerOneNeuronA2 + weightToOutputNeuronY1FromLayerOneNeuronA2Update
		firstLayerNeuronA1Bias = firstLayerNeuronA1Bias + firstLayerNeuronA1BiasUpdate
		firstLayerNeuronA2Bias = firstLayerNeuronA2Bias + firstLayerNeuronA2BiasUpdate
		outputLayerNeuronY1Bias = outputLayerNeuronY1Bias + outputLayerNeuronY1BiasUpdate

	elif(traintype == 'batch'):
		batchsize = 4
		batcherrors = []

		errorGradientWRTweightToOutputNeuronY1FromLayerOneNeuronA1 = 0
		errorGradientWRTweightToOutputNeuronY1FromLayerOneNeuronA2 = 0

		errorGradientWRTweightToLayerOneNeuronA1FromInputNeuronX1 = 0
		errorGradientWRTweightToLayerOneNeuronA1FromInputNeuronX2 = 0

		errorGradientWRTweightToLayerOneNeuronA2FromInputNeuronX1 = 0
		errorGradientWRTweightToLayerOneNeuronA2FromInputNeuronX2 = 0

		errorGradientWRTfirstLayerNeuronA1Bias = 0
		errorGradientWRTfirstLayerNeuronA2Bias = 0

		errorGradientWRToutputLayerNeuronY1Bias = 0

		trainingpairs = [[0,0],[0,1],[1,0],[1,1]]
		random.shuffle(trainingpairs)


		for k in range(batchsize):
			nodeDeltaLayerOneNeuronA1 = 0
			nodeDeltaLayerOneNeuronA2 = 0
			nodeDeltaOutputNeuronY1 = 0



			# Forward Prop
			x1Value = trainingpairs[k%4][0]
			x2Value = trainingpairs[k%4][1]
			actualValue = feedForwardPropagation(x1Value,x2Value)

			idealValue = XORfunction(x1Value,x2Value)

			error = errorValue(actualValue, idealValue)
			batcherrors.append(error)
			# print(error)
			errors.append(abs(error))


			derivative = sigmoidDerivative(outputNeuronY1Activation)

			nodeDeltaOutputNeuronY1 = nodeDelta(error, derivative, 'outputNeuron',[],[])


			weightsFromLayerOneNeuronA1 = [weightToOutputNeuronY1FromLayerOneNeuronA1]
			deltasFromLayerOneNeuronA1 = [nodeDeltaOutputNeuronY1]
			derivative = sigmoidDerivative(firstLayerNeuronA1Activation)
			nodeDeltaLayerOneNeuronA1 = nodeDelta(error, derivative, 'interiorNeuron', deltasFromLayerOneNeuronA1, weightsFromLayerOneNeuronA1)

			weightsFromLayerOneNeuronA2 = [weightToOutputNeuronY1FromLayerOneNeuronA2]
			deltasFromLayerOneNeuronA2 = [nodeDeltaOutputNeuronY1]
			derivative = sigmoidDerivative(firstLayerNeuronA2Activation)
			nodeDeltaLayerOneNeuronA2 = nodeDelta(error, derivative, 'interiorNeuron', deltasFromLayerOneNeuronA2, weightsFromLayerOneNeuronA2)

			# Gradients

			errorGradientWRTweightToOutputNeuronY1FromLayerOneNeuronA1 = firstLayerNeuronA1Activation*nodeDeltaOutputNeuronY1 + errorGradientWRTweightToOutputNeuronY1FromLayerOneNeuronA1
			errorGradientWRTweightToOutputNeuronY1FromLayerOneNeuronA2 = firstLayerNeuronA2Activation*nodeDeltaOutputNeuronY1 + errorGradientWRTweightToOutputNeuronY1FromLayerOneNeuronA2

			errorGradientWRTweightToLayerOneNeuronA1FromInputNeuronX1 = inputNeuronX1*nodeDeltaLayerOneNeuronA1 + errorGradientWRTweightToLayerOneNeuronA1FromInputNeuronX1
			errorGradientWRTweightToLayerOneNeuronA1FromInputNeuronX2 = inputNeuronX2*nodeDeltaLayerOneNeuronA1 + errorGradientWRTweightToLayerOneNeuronA1FromInputNeuronX2 

			errorGradientWRTweightToLayerOneNeuronA2FromInputNeuronX1 = inputNeuronX1*nodeDeltaLayerOneNeuronA2 + errorGradientWRTweightToLayerOneNeuronA2FromInputNeuronX1
			errorGradientWRTweightToLayerOneNeuronA2FromInputNeuronX2 = inputNeuronX2*nodeDeltaLayerOneNeuronA2 + errorGradientWRTweightToLayerOneNeuronA2FromInputNeuronX2

			errorGradientWRTfirstLayerNeuronA1Bias = firstLayerNeuronA1Bias*nodeDeltaLayerOneNeuronA1 + errorGradientWRTfirstLayerNeuronA1Bias
			errorGradientWRTfirstLayerNeuronA2Bias = firstLayerNeuronA2Bias*nodeDeltaLayerOneNeuronA2 + errorGradientWRTfirstLayerNeuronA2Bias

			errorGradientWRToutputLayerNeuronY1Bias = outputLayerNeuronY1Bias*nodeDeltaOutputNeuronY1 + errorGradientWRToutputLayerNeuronY1Bias


		

		weightToLayerOneNeuronA1FromInputNeuronX1Update = (epsilon*errorGradientWRTweightToLayerOneNeuronA1FromInputNeuronX1) + (alpha*weightToLayerOneNeuronA1FromInputNeuronX1Update)
		weightToLayerOneNeuronA1FromInputNeuronX2Update = (epsilon*errorGradientWRTweightToLayerOneNeuronA1FromInputNeuronX2) + (alpha*weightToLayerOneNeuronA1FromInputNeuronX2Update)
		weightToLayerOneNeuronA2FromInputNeuronX1Update = (epsilon*errorGradientWRTweightToLayerOneNeuronA2FromInputNeuronX1) + (alpha*weightToLayerOneNeuronA2FromInputNeuronX1Update)
		weightToLayerOneNeuronA2FromInputNeuronX2Update = (epsilon*errorGradientWRTweightToLayerOneNeuronA2FromInputNeuronX2) + (alpha*weightToLayerOneNeuronA2FromInputNeuronX2Update)
		weightToOutputNeuronY1FromLayerOneNeuronA1Update = (epsilon*errorGradientWRTweightToOutputNeuronY1FromLayerOneNeuronA1) + (alpha*weightToOutputNeuronY1FromLayerOneNeuronA1Update)
		weightToOutputNeuronY1FromLayerOneNeuronA2Update = (epsilon*errorGradientWRTweightToOutputNeuronY1FromLayerOneNeuronA2) + (alpha*weightToOutputNeuronY1FromLayerOneNeuronA2Update)
		firstLayerNeuronA1BiasUpdate = (epsilon*errorGradientWRTfirstLayerNeuronA1Bias) + (alpha*firstLayerNeuronA1BiasUpdate)
		firstLayerNeuronA2BiasUpdate = (epsilon*errorGradientWRTfirstLayerNeuronA2Bias) + (alpha*firstLayerNeuronA2BiasUpdate)
		outputLayerNeuronY1BiasUpdate = (epsilon*errorGradientWRToutputLayerNeuronY1Bias) + (alpha*outputLayerNeuronY1BiasUpdate)
		
		weightToLayerOneNeuronA1FromInputNeuronX1 = weightToLayerOneNeuronA1FromInputNeuronX1 + weightToLayerOneNeuronA1FromInputNeuronX1Update
		weightToLayerOneNeuronA1FromInputNeuronX2 = weightToLayerOneNeuronA1FromInputNeuronX2 + weightToLayerOneNeuronA1FromInputNeuronX2Update
		weightToLayerOneNeuronA2FromInputNeuronX1 = weightToLayerOneNeuronA2FromInputNeuronX1 + weightToLayerOneNeuronA2FromInputNeuronX1Update
		weightToLayerOneNeuronA2FromInputNeuronX2 = weightToLayerOneNeuronA2FromInputNeuronX2 + weightToLayerOneNeuronA2FromInputNeuronX2Update
		weightToOutputNeuronY1FromLayerOneNeuronA1 = weightToOutputNeuronY1FromLayerOneNeuronA1 + weightToOutputNeuronY1FromLayerOneNeuronA1Update
		weightToOutputNeuronY1FromLayerOneNeuronA2 = weightToOutputNeuronY1FromLayerOneNeuronA2 + weightToOutputNeuronY1FromLayerOneNeuronA2Update
		firstLayerNeuronA1Bias = firstLayerNeuronA1Bias + firstLayerNeuronA1BiasUpdate
		firstLayerNeuronA2Bias = firstLayerNeuronA2Bias + firstLayerNeuronA2BiasUpdate
		outputLayerNeuronY1Bias = outputLayerNeuronY1Bias + outputLayerNeuronY1BiasUpdate

		# error = sum(batcherrors)/float(len(batcherrors))
		# print(error)
	return error

errors = []
# itercount = []

epsilon = 0.001
alpha = 0.1
	
for i in range(100000):
	er = backPropagation('batch')
	if i % 1000 == 0:
		print(er)
	# errors.append(abs(er))
	# itercount.append(i)

trainingpairs = [[0,0],[0,1],[1,0],[1,1]]
for i in range(4):
	x1Value = trainingpairs[i%4][0]
	x2Value = trainingpairs[i%4][1]
	actualValue = feedForwardPropagation(x1Value,x2Value)


	idealValue = XORfunction(x1Value,x2Value)
	print("The Output of the Feed Foward Propagation with inputs \n%f to input Neuron X1 and %f to input Neuron X2 is %f. Ideal Value is %d." % (x1Value,x2Value,actualValue,idealValue))

	error = errorValue(actualValue, idealValue)



plt.plot(errors)
plt.show()



