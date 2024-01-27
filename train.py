import json, numpy as np
from nltk_utils import tokenize, stem, bagOfWords

import torch, torch.nn as nn
from torch.utils.data import Dataset,DataLoader

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

dataset = ChatDataset()
trainloader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=2)
