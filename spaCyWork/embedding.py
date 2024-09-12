import spacy
import numpy as np
from spacy.tokens import DocBin, Doc    
from os.path import isfile
from spacy.vocab import Vocab
import json
from tools import urlToFilename

VOCAB_FOLDER = 'spaCyWork/Data/SpacyData/Vocab' #create my own folder
DOCBIN_FOLDER = 'spaCyWork/Data/SpacyData/DocBins'
IN_FILE = 'spaCyWork/Data/allArticles.txt'
N_THREADS = 8
ARTICLE_LIMIT = 10
VOCAB = Vocab().from_disk("spaCyWork/Data/SpacyData/Vocab")
start = 0
stop = 10



nlp = spacy.load("en_core_web_md")


def getArticleDocBin(articleKey):
    fpath = articleKey
    if not isfile(DOCBIN_FOLDER + "/" + fpath):
        raise ValueError(f'Docbin not found for article with key: {articleKey}')

    dbin = DocBin().from_disk(DOCBIN_FOLDER + "/"  + fpath)

    return dbin





def loadArticles(file=IN_FILE, maxYields=ARTICLE_LIMIT):
    n = 0
    with open(file, 'r') as f:
        for line in f:
            if n >= maxYields:
                break

            article = json.loads(line.strip())
            yield article  # yield the article and pause
            n += 1  


def getArticles(generator):
    for article in generator:
        key = article['url']
        toFile = urlToFilename(key)
        docList = list(getArticleDocBin(toFile).get_docs(VOCAB))
        
        #Combine the processed docs into a list and get their values (i.e., their doc objects)
        try:
            assert docList is not None
            assert len(docList) == len(article['content'])
        except AssertionError as e:
            print(f'Docbin error: {key}')
            continue

        

    
        

def getSpanText(docs, start, stop):
    text = []
    vectors = []

    for doc in docs: # words in paragraphs
        for token in doc: # tokens in word
            if len(text) >= stop:
                break
            if len(text) >= start:
                text.append(token.text)
                vectors.append(token.vector)


    # doing both by tokens rn? 


    segmentText = " ".join(text[start:stop])
    #print("Segment Text:", segmentText)

    segmentVectors = vectors[start:stop]
    #print("Segment Vectors:", segmentVectors)

    return segmentText, segmentVectors

# split one method for text and then one for vectors
# fix nlp.pipe
# doc = Doc.from_docs(docList[start:stop]) # check the method's documentation
# # 


def main():
        articles = loadArticles(IN_FILE, ARTICLE_LIMIT)
        docs = getArticles(articles)

        #extract text and vectors from a specific span, for example from index 0 to 10

        #output the results
        #print(f"Extracted Text: {segmentText}")
        #print(f"Vectors: {np.array(segmentVectors)}")\
        

if __name__ == "__main__":
    main()
