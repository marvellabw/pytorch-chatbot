import nltk, numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
	return nltk.word_tokenize(sentence)

def stem(word):
	return stemmer.stem(word.lower())

def bagOfWords(tokenizedSentence, allWords):
	tokenizedSentence = [stem(w) for w in tokenizedSentence]
	bag = np.zeros(len(allWords), dtype=np.float32)

	for index, word in enumerate(allWords):
		if word in tokenizedSentence:
			bag[index] = 1.0
	
	return bag