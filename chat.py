import random
import json
import torch

from model import NeuralNet
from nltk_utils import bagOfWords, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as file:
	intents = json.load(file)

FILE = "data.pth"
data = torch.load(FILE)

inputSize = data["inputSize"]
hiddenSize = data["hiddenSize"]
outputSize = data["outputSize"]
allWords = data["allWords"]
tags = data["tags"]
modelState = data["modelState"]

model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)
model.load_state_dict(modelState)
model.eval()

botName = "BowoBot"
print("Let's chat! type 'quit' to exit")

while True:
	sentence = input('You:  ')
	if sentence == "quit":
		break

	sentence = tokenize(sentence)
	x = bagOfWords(sentence, allWords)
	x = x.reshape(1, x.shape[0])
	x = torch.from_numpy(x).to(device)

	output = model(x)
	_, predicted = torch.max(output, dim=1)
	tag = tags[predicted.item()]

	probs = torch.softmax(output, dim=1)
	prob = probs[0][predicted.item()]

	if prob.item() > 0.75:
		for intent in intents['intents']:
			if tag == intent['tag']:
				print(f"{botName}:  {random.choice(intent['responses'])}")
	else:
		print(f"{botName}:  Sorry, I did not understand that..")