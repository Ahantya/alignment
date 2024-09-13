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
VOCAB = Vocab().from_disk(VOCAB_FOLDER)
start = 0
stop = 2



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
        docList = list(getArticleDocBin(toFile).get_docs(nlp.vocab)) #nlp.vocab works
        
        #Combine the processed docs into a list and get their values (i.e., their doc objects)
        try:
            assert docList is not None
            assert len(docList) == len(article['content'])
        except AssertionError as e:
            print(f'Docbin error: {key}')
            continue

        segmentTest = getSpanText(docList)
        vectorTest = getVectors(docList)
        



def getVectors(docs):
    combined = Doc.from_docs(docs)
    wordRange = combined[start:stop]
    vectors = []
    for token in wordRange:
            if token.has_vector:
                vectors.append(token.vector)
            else:
                #print(f"Token '{token.text}' does not have a vector.")
                continue
    return np.array(vectors)


    
        

def getSpanText(docs):

    combined = Doc.from_docs(docs)
    wordRange = combined[start:stop]

    spanText = " ".join([token.text for token in wordRange])

    return spanText
    


# split one method for text and then one for vectors DONE
# fix nlp.pipe DONE
# doc = Doc.from_docs(docList[start:stop]) # check the method's documentation DONE
# fix whitespace to be natural tokenization or something 
# token 2 and 3 are like the 2nd and 3rd word of the combined paragraphs DONE


def main():
        articles = loadArticles(IN_FILE, ARTICLE_LIMIT)
        getArticles(articles)


        #extract text and vectors from a specific span, for example from index 0 to 10

        #output the results
        #print(f"Extracted Text: {segmentText}")
        #print(f"Vectors: {np.array(segmentVectors)}")\


        # Assuming you already have a DocBin saved, load it and inspect

        
        

if __name__ == "__main__":
    main()
