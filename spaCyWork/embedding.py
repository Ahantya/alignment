import spacy
import numpy as np
from spacy.tokens import DocBin, Doc    
from os.path import isfile
from spacy.vocab import Vocab
import json
from spacy.training import Alignment
from tools import urlToFilename

VOCAB_FOLDER = 'spaCyWork/Data/SpacyData/Vocab' #create my own folder
DOCBIN_FOLDER = 'spaCyWork/Data/SpacyData/DocBins'
IN_FILE = 'spaCyWork/Data/allArticles.txt'
IN_FILE = 'spaCyWork/Data/newArticle.txt'
N_THREADS = 8
ARTICLE_LIMIT = 10
VOCAB = Vocab().from_disk(VOCAB_FOLDER)


# use sublime to open allArticles.txt
# manually look at space range (stop is exclusive)



nlp = spacy.load("en_core_web_md")

def spaceRangeToSpacyRange(start, stop, articleDoc):
    spaceTokens = articleDoc.text.split(" ") 
    spacyTokens = [token.text for token in articleDoc]
    align = Alignment.from_strings(spaceTokens, spacyTokens)
    spacyStart = align.x2y[start][0]
    spacyEnd = align.x2y[stop][-1] 
    return (spacyStart, spacyEnd)


# TEST THIS

def getArticleDocBin(articleKey):
    fpath = articleKey
    if not isfile(DOCBIN_FOLDER + "/" + fpath):
        raise ValueError(f'Docbin not found for article with key: {articleKey}')

    dbin = DocBin().from_disk(DOCBIN_FOLDER + "/"  + fpath)

    return dbin


def createDocsFromArticle(article):
    #get content and split into paragraphs
    content = article['content'] 
    paragraphs = content.split('\n') 

    docList = []
    for para in paragraphs:
        # Tokenize the paragraph into a list of words (or you can use sentence segmentation)
        tokens = para.split()

        doc = spacy.tokens.Doc(nlp.vocab, words=tokens)

        docList.append(doc)

    return docList


def loadArticles(file=IN_FILE, maxYields=ARTICLE_LIMIT):
    n = 0
    with open(file, 'r') as f:
        for line in f:
            if n >= maxYields:
                break

            article = json.loads(line.strip())
            
            yield article  # yield the article and pause
            n += 1  


def getDocList(article):
        key = article['url']
        toFile = urlToFilename(key)
        docList = list(getArticleDocBin(toFile).get_docs(nlp.vocab)) #nlp.vocab works
        
        #Combine the processed docs into a list and get their values (i.e., their doc objects)
        try:
            assert docList is not None
            assert len(docList) == len(article['content'])
            return docList
        except AssertionError as e:
            print(f'Docbin error: {key}')
            return
        



def getSpanVectors(articleDoc, start, stop):
    wordRange = articleDoc[start:stop]
    vectors = []
    for token in wordRange:
            if token.has_vector:
                vectors.append(token.vector)
            else:
                #print(f"Token '{token.text}' does not have a vector.")
                continue
    return np.array(vectors)


        

def getSpanText(articleDoc, start, stop):
    wordRange = articleDoc[start:stop]

    spanText = " ".join([token.text for token in wordRange])

    return spanText

# should we be getting a list of docs or a single doc? 
    


# split one method for text and then one for vectors DONE
# fix nlp.pipe DONE
# doc = Doc.from_docs(docList[start:stop]) # check the method's documentation DONE
# fix whitespace to be natural tokenization or something 
# token 2 and 3 are like the 2nd and 3rd word of the combined paragraphs DONE

# 
def main():
        articleGenerator = loadArticles(IN_FILE, ARTICLE_LIMIT)
        #print(list(articleGenerator))
        # count = 0
        # for article in articleGenerator:
        #     count+= 1
        # print(count)
        start = 0 #sends
        stop = 2 #before driving

        #start = 15 #driving 
        #stop = 20 #before and 

        #start = 350 
        #stop = 354 # last index


        # this is working 
        for article in articleGenerator:
             docList = createDocsFromArticle(article)
             #docList = getDocList(article)
             combinedDoc = Doc.from_docs(docList)
            #  # one Doc per article
             (spacyStart, spacyStop) = spaceRangeToSpacyRange(start, stop, combinedDoc)
             segmentText = getSpanText(combinedDoc, spacyStart, spacyStop)
             print(segmentText) # manually verify that this is the same as the spaces we looked at the article (check above comments)
             exit() #stops after first article
             spaceText = " ".join(combinedDoc.text.split(" ")[start:stop])
             (spacyStart, spacyStop) = spaceRangeToSpacyRange(start, stop, combinedDoc)
             spacyText = getSpanText(combinedDoc, spacyStart, spacyStop)
             assert spaceText == spacyText


        #extract text and vectors from a specific span, for example from index 0 to 10

        #output the results
        #print(f"Extracted Text: {segmentText}")
        #print(f"Vectors: {np.array(segmentVectors)}")\


        # Assuming you already have a DocBin saved, load it and inspect

        
        

if __name__ == "__main__":
    main()

    # on articles.txt, article content should be one string (manually do first one)
    # use newline characters (enter)
    # modify the runSpacy # look at comments on file
    # test 3 cases (1. all contained in first paragraph, 2. all in another paragraph, 3. text is across both paragraphs) 
    # make a new article so i can test



