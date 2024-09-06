from spacy.tokens import DocBin, Doc
from spacy.vocab import Vocab
from os.path import isfile
from math import pow, log2
import json
import re
DOCBIN_FOLDER = './Data/SpacyData/DocBins'
VOCAB_FOLDER = "./Data/SpacyData/Vocab"
VOCAB = None


def urlToFilename(url):
    urlPattern = r"^https\:\/\/www\.cnn\.com\/(.+)\/index\.html$"
    m = re.match(urlPattern,url)

    if m is None:
        return 'UNK'

    name = m.group(1).replace('/','_')

    return name

def getURLCategory(url):
    import re
    urlPattern = r"^https://www\.cnn\.com/[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/(.+?)/(?:.+?)/index\.html$"
    altPattern = r"^https://www\.cnn\.com/(.+?)/article/(?:.+?)/index\.html$"
    livePattern = r"^https://www\.cnn\.com/(.+?)/live-news/(?:.+?)/index\.html$"

    m = re.match(urlPattern,url) or re.match(altPattern,url) or re.match(livePattern,url)

    if m is None:
        return 'UNK'

    category = m.group(1) + ('' if re.match(livePattern,url) is None else '-live-news')

    return category

def getHTML(url,wait=0):
    import requests
    import time

    time.sleep(wait)

    try:
        res = requests.get(url)

        res.raise_for_status()

        html = res.text
    except requests.exceptions.RequestException as e:
        raise

    return html

def urlBatchGenerator(urlFileName, batches = 8, jobLimit=125):
    from math import floor
    try:
        with open(urlFileName,'r') as f:
            
            lines = [line.strip() for line in f.readlines()]

            urls = lines[:jobLimit]

            print(f'{len(urls)} urls to process.')

            jobs = len(urls)
            jobSize = floor(jobs/batches)
            extraJobs = jobs - (jobSize*batches)

            for i in range(batches):

                size = jobSize
                if i+1 <= extraJobs:
                    size+=1

                batch = urls[:size]
                urls = urls[size:]

                yield batch

            assert len(urls) == 0

    except OSError as e:
        raise


def getArticleDocBinList(articleKey):
    global VOCAB
    fpath = DOCBIN_FOLDER + '/' + urlToFilename(articleKey)
    if not isfile(fpath):
        raise ValueError(f'Docbin not found for article with key: {articleKey}')

    dbin = DocBin().from_disk(fpath)

    if VOCAB is None:
        VOCAB = Vocab().from_disk(VOCAB_FOLDER)

    docs = list(dbin.get_docs(VOCAB))

    return docs

STOPWORDS = {'i','it','is','be','can','could','should','not', 'because','have','had', 'if','then','would','the','that','those','these','this','us','we','by','than','an','at','oh','about','as',
'_','mm','hmm','with','-','or','this','on','in','of','so','um','a','and','to','uh',',','.','?','!',';','"',"'"}

def keepToken(token):
    return (not (token.like_num or token.is_quote or token.is_bracket or token.is_space or token.is_punct or token.is_oov) 
        and not (token.text.lower() in STOPWORDS or token.lemma_.lower() in STOPWORDS)
        and (token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'PROPN' or token.pos_ == 'ADJ' or token.pos_ == 'ADV'))


def getBagOfWords(articleKey,start,stop):
    docs = getArticleDocBinList(articleKey)
    segDoc = Doc.from_docs(docs[start:stop])
    if segDoc is None:
        print('wtf')
    lemmas = {token.lemma_ for token in segDoc if keepToken(token)}

    return lemmas


def getSents(articleKey,start,stop):
    docs = getArticleDocBinList(articleKey)[start:stop]
    sents = []
    for parDoc in docs:
        for sent in parDoc.sents:
            tmp = []
            for token in sent:
                if keepToken(token):
                    tmp.append(token)
            if tmp:
                sents.append(tmp)

    return sents

def getPars(articleKey,start,stop):
    docs = getArticleDocBinList(articleKey)[start:stop]
    sents = []
    for parDoc in docs:
        for sent in parDoc.sents:
            tmp = []
            for token in sent:
                if keepToken(token):
                    tmp.append(token)
            if tmp:
                sents.append(tmp)

    return sents

def getTokens(articleKey,start,stop):
    docs = getArticleDocBinList(articleKey)[start:stop]
    res = []
    for parDoc in docs:
        for sent in parDoc.sents:
            tmp = []
            for token in sent:
                if keepToken(token):
                    res.append(token)

    return res

def getTokenSimilarity(tokenA,tokenB):
    if tokenA.text.lower().strip()==tokenB.text.lower().strip():
        return 1
    if tokenA.lemma_==tokenB.lemma_:
        return 1

    if not tokenA.has_vector or not tokenB.has_vector:
        print(f'{tokenA.text} - {tokenB.text}')
        return 0

    return tokenA.similarity(tokenB)


def getLemmaWeights(articleKey,start,stop):
    docs = getArticleDocBinList(articleKey)

    outCounts = {}
    segCounts = {}
    vocab = set()
    n = 0
    for i, doc in enumerate(docs):
        countDict =  outCounts if i < start or i>= stop else segCounts

        tokens = filter(lambda t: keepToken(t),doc)
        for t in tokens:
            n+=1
            lemma = t.lemma_
            vocab.add(lemma)
            if lemma not in countDict:
                countDict[lemma] = 0
            countDict[lemma] += 1
    for w in vocab:
        outCounts[w] = outCounts[w] if w in outCounts else 0
        segCounts[w] = segCounts[w] if w in segCounts else 0

    pmiDict = {}
    for w in vocab:
        probW = (outCounts[w]+segCounts[w])/n
        segObs = max(0,sum(segCounts.values()))
        condProbW = segCounts[w]/segObs
        pmi = log2(condProbW) - log2(probW)
        pmiDict[w] = pmi/(-(log2(probW)+log2()))

    vals = list(pmiDict.values())
    small = min(vals)
    big = max(vals)

    for w in pmiDict:
        pmi = pmiDict[w]
        pmiDict[w] = (pmi-small)/(big-small)



    if segDoc is None:
        print('wtf')
    lemmas = {token.lemma_ for token in segDoc if keepToken(token)}
    tokens = []
    seenLemmas = set()
    for token in segDoc:
        if keepToken(token) and token.lemma_ not in seenLemmas:
            tokens.append(token)
            seenLemmas.add(token.lemma_)

    return tokens


def getArticleList(DS_FILE, jobLimit):
    n = 0
    with open(DS_FILE,'r',encoding='utf-8') as i:
        for line in i:
            article = json.loads(line.strip())
            n+=1
            yield article
            if (jobLimit!=None) and n>=jobLimit:
                    break


