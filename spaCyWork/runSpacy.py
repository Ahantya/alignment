import time
import logging
import spacy
import json
from datetime import timedelta
from spacy.tokens import DocBin
from spacy.tokens import Doc
from os.path import isfile
from math import inf
from tools import urlToFilename
import os

VOCAB_FOLDER = 'spaCyWork/Data/SpacyData/Vocab' #create my own folder
DOCBIN_FOLDER = 'spaCyWork/Data/SpacyData/DocBins'
IN_FILE = 'spaCyWork/Data/allArticles.txt'
N_THREADS = 8
ARTICLE_LIMIT = 10


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
           spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)
	
# nlp = spacy.blank("en")
# nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
# doc = nlp("What's happened to me? he thought. It wasn't a dream.")
# print([token.text for token in doc])



def articleGenerator(file=IN_FILE,maxYields=ARTICLE_LIMIT):
	n = 0
	with open(file,'r') as f:
		for line in f:
			if n >= maxYields:
				break

			article = json.loads(line.strip())
			key = article['url']

			if (not isfile(urlToFilename(key))): #once spacy processes an article, checks for processing
				n+=1
				yield article # return and pause

def getSpacyDocs(nlp,generator):

	texts = []
	docDict = {}
	for article in generator:
		docDict[article['url']] = [None]*len(article['content'])

		for i,par in enumerate(article['content']):
			texts.append((par,{'url': article['url'], 'index': i}))

	#print(f'Processing {len(docDict)} articles...')
	print("who?")

	doc_tuples = nlp.pipe(texts, n_process=N_THREADS, as_tuples=True)

	for doc, context in doc_tuples:
		docDict[context['url']][context['index']] = doc

	print("ok")

	return docDict



def run ():

	nlp = spacy.load("en_core_web_sm")
	nlp.disable_pipe("parser")
	nlp.enable_pipe("senter")

	nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


	articleGen = articleGenerator(IN_FILE)

	docDict = getSpacyDocs(nlp,articleGen)

	if not os.path.exists(DOCBIN_FOLDER):
		os.makedirs(DOCBIN_FOLDER)

	nlp.vocab.to_disk(VOCAB_FOLDER)

	for key,docList in docDict.items():
		fpath = DOCBIN_FOLDER + '/' + urlToFilename(key)
		dbin = DocBin(docs=docList)
		dbin.to_disk(fpath)
	# make above into a function	

	#print ('Finished text processing')



if __name__ == "__main__":
	run()


# do whitespace tokenizer before article generator