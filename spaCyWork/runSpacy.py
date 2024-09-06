import time
import logging
import spacy
import json
from datetime import timedelta
from spacy.tokens import DocBin
from os.path import isfile
from math import inf
from tools import urlToFilename

VOCAB_FOLDER = './Data/SpacyData/Vocab' #create my own folder
DOCBIN_FOLDER = './Data/SpacyData/DocBins'
IN_FILE = './Data/allArticles.txt'
N_THREADS = 8
ARTICLE_LIMIT = inf



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

	print(f'Processing {len(docDict)} articles...')

	doc_tuples = nlp.pipe(texts, n_process=N_THREADS, as_tuples=True)

	for doc, context in doc_tuples:
		docDict[context['url']][context['index']] = doc

	return docDict



def run ():

	nlp = spacy.load("en_core_web_lg")
	nlp.disable_pipe("parser")
	nlp.enable_pipe("senter")

	logger.info("Loaded spacy...")

	articleGen = articleGenerator(IN_FILE)

	docDict = getSpacyDocs(nlp,articleGen)

	nlp.vocab.to_disk(VOCAB_FOLDER)

	for key,docList in docDict.items():
		fpath = DOCBIN_FOLDER + '/' + urlToFilename(key)
		dbin = DocBin(docs=docList)
		dbin.to_disk(fpath)
	# make above into a function	

	#print ('Finished text processing')



if __name__ == "__main__":
    start = time.time()
    logging.basicConfig(level="INFO", handlers=[
        logging.FileHandler("spacy.log", mode='w'),
        logging.StreamHandler()
    ])
    logger = logging.getLogger(__name__)
    logger.info(f"Start:{time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(start))}")
    run()
    end = time.time()
    logger.info(f"Run Time: {timedelta(seconds=end-start)}")


# do whitespace tokenizer before article generator