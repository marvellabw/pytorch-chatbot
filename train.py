import json, numpy as np
from nltk_utils import tokenize, stem, bagOfWords

import torch, torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from model import NeuralNet

with open('intents.json', 'r') as file:
	intents = json.load(file)

allWords = []
tags = []
xy = [] 

for intent in intents['intents']:
	tag = intent['tag']
	tags.append(tag)

	for pattern in intent['patterns']:
		tokenizedPattern = tokenize(pattern)
		allWords.extend(tokenizedPattern)
		xy.append((tokenizedPattern, tag))

punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
allWords = [stem(word) for word in allWords if word not in punctuation]

allWords = sorted(set(allWords))
tags = sorted(set(tags))

xTrain = []
yTrain = []

for (patternSentence, tag) in xy:
	bag = bagOfWords(patternSentence, allWords)
	xTrain.append(bag)

	label = tags.index(tag)
	yTrain.append(label) # CrossEntropyLoss

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

class ChatDataset(Dataset):
	def __init__(self) -> None:
		self.nSamples = len(xTrain)
		self.xData = xTrain
		self.yData = yTrain
	
	def __getitem__(self, index) -> Any:
		return self.xData[index], self.yData[index]
	
	def __len__(self):
		return self.nSamples


batchSize = 8
inputSize = len(xTrain[0])
hiddenSize = 8
outputSize = len(tags)
learningRate = 0.001
numEpochs = 1000

dataset = ChatDataset()
trainloader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

for epoch in range(numEpochs):
	for (words, labels) in trainloader:
		words = words.to(device)
		labels = labels.to(device)

		outputs = model(words)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	if (epoch + 1) % 100 == 0:
		print(f'epoch {epoch+1}/{numEpochs}, loss={loss.item():.4f}')

print(f'final loss: loss={loss.item():.4f}')