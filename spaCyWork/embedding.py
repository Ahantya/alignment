import spacy
import numpy as np
from spacy.tokens import DocBin, Doc    
import json

VOCAB_FOLDER = 'spaCyWork/Data/SpacyData/Vocab' #create my own folder
DOCBIN_FOLDER = 'spaCyWork/Data/SpacyData/DocBins'
IN_FILE = 'spaCyWork/Data/allArticles.txt'
N_THREADS = 8
ARTICLE_LIMIT = 100


nlp = spacy.load("en_core_web_md")


def loadArticles(file=IN_FILE, maxYields=ARTICLE_LIMIT):
    """
    Generator that yields articles one by one from the input file.

    Args:
    - file: Path to the file containing the articles.
    - maxYields: Maximum number of articles to yield.

    Yields:
    - article: A json file representing an article.
    """
    n = 0
    with open(file, 'r') as f:
        for line in f:
            if n >= maxYields:
                break

            article = json.loads(line.strip())
            yield article  # yield the article and pause
            n += 1  

def processArticle(article, nlp):
    """
    Process a single article with spaCy and return the Doc objects.

    Args:
    - article: Dictionary representing an article.
    - nlp: spaCy pipeline.

    Returns:
    - docDict: Dictionary of processed Doc objects keyed by their index in the article.
    """
    docDict = {}
    texts = [(par, {'url': article['url'], 'index': i}) for i, par in enumerate(article['content'])]

    # Process texts with spaCy pipeline
    docTuples = nlp.pipe(texts, n_process=N_THREADS, as_tuples=True)
    
    for doc, context in docTuples:
        docDict[context['index']] = doc
        # index of paragraph, so it stores each paragraph in doc separately

    return docDict

def getSpanText(docs, start, stop):
    """
    Extract and return text and vectors for a specific range from a list of Docs.
    
    Args:
    - docs: List of spacy.tokens.Doc objects.
    - start: Start index (inclusive).
    - stop: End index (exclusive).
    
    Returns:
    - Tuple of (segmentText, segmentVectors) where:
        - segmentText: List of words in the specified range.
        - segmentvectors: List of word vectors in the specified range.
    """
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






def main():
    articles = loadArticles(IN_FILE, ARTICLE_LIMIT)

    for article in articles:
        print(f"Processing article from URL: {article['url']}")
        processedDocs = processArticle(article, nlp)
        
        #combine the processed docs into a list and get their values (ie. their doc objects)
        docs = list(processedDocs.values())

        #extract text and vectors from a specific span, for example from index 0 to 10
        segmentText, segmentVectors = getSpanText(docs, 0, 10)

        #output the results
        print(f"Extracted Text: {segmentText}")
        print(f"Vectors: {np.array(segmentVectors)}")\
        
        break

if __name__ == "__main__":
    main()
