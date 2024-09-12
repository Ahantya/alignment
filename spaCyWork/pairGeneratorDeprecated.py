
# import time
# from datetime import timedelta
# import logging
# import ray
# import json
# from spacy.tokens import DocBin, Doc
# from spacy.vocab import Vocab
# from os.path import isfile
# from math import log2, sqrt

# STOPWORDS = {'i','it','the','that','those','these','this','us','we','by','than','an','at','oh','about','as',
# '_','mm','hmm','with','-','or','this','on','in','of','so','um','a','and','to','uh',',','.','?','!',';','"',"'"}
# DATASET_FILE = './data/politicsDump.txt'
# MAX_SEGS=6
# MIN_SEGS=4
# VOCAB = Vocab().from_disk(".Data/SpacyData/vocab")

# def keepToken(token):
#     return (not (token.like_num or token.is_quote or token.is_bracket or token.is_space or token.is_punct or token.is_oov) 
#     	and not (token.text.lower() in STOPWORDS or token.lemma_.lower() in STOPWORDS) 
#     	and (token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'PROPN' ))

# def isSelected(article):
# 	n = len(article['segmentation'])

# 	if n<MIN_SEGS or n>MAX_SEGS:
# 		#print(f'f1: {n}; {article["key"]}')
# 		return False

# 	MIN_SEG_PARS = 2
# 	MIN_SEG_LEN = 75
# 	MIN_PAR_LEN = 50
# 	for (start,stop,_) in article['segmentation']:
# 		nPars = stop-start
		
# 		if nPars<MIN_SEG_PARS:
# 			#print(f'f2: {nPars}: {article["key"]}')
# 			return False

# 		segLen = 0
# 		pars = article['content'][start:stop]
# 		for p in pars:
# 			l = len(p.split())

# 			segLen += l

# 		if segLen < MIN_SEG_LEN:
# 			#print(f'f4: {segLen}: {article["key"]}')
# 			return False

# 	return True




# def articleGenerator(file):
# 	n = 0
# 	with open(file,'r') as f:
# 		for line in f:
# 			article = json.loads(line.strip())
# 			key = article['key']

# 			if isSelected(article):
# 				yield article


# def getDocBinPathFromKey(articleKey):
# 	return f"{DOCBIN_FOLDER}/{articleKey.split('/')[-2]}.spacy"

# def getArticleDocBin(articleKey):
# 	fpath = getDocBinPathFromKey(articleKey)
# 	if not isfile(fpath):
# 		raise ValueError(f'Docbin not found for article with key: {articleKey}')

# 	dbin = DocBin().from_disk(fpath)

# 	return dbin




# def getLemmaProbs(doc):

# 	counts = {}
# 	n = 0
# 	for token in doc:
# 		if not keepToken(token):
# 			continue

# 		lemma = token.lemma_
# 		n+=1

# 		if lemma not in counts:
# 			counts[lemma] = 0

# 		counts[lemma] += 1

# 	for lemma in counts:
# 		counts[lemma] = counts[lemma]/n

# 	return counts



# def updateDict(tallyDict,nextDict,op,nPrevUp=0):
# 	if op not in {'acc','app'}:
# 		raise ValueError('invalid merge type')

# 	keys = tallyDict.keys() | nextDict.keys()
# 	for k in keys:
# 		if op=='acc':
# 			tally = tallyDict[k] if k in tallyDict else 0
# 			r = tally + (nextDict[k] if k in nextDict else 0)
# 		elif op=='app':
# 			tally = tallyDict[k] if k in tallyDict else [0 for x in range(nPrevUp)]
# 			r = tally + [nextDict[k] if k in nextDict else 0]
		
# 		tallyDict[k] = r



# def getNPMI(wordProbs,wordCondProbs,nTopics):
# 	pT = 1/nTopics

# 	npmiDict = {}
# 	for w,pW in wordProbs.items():
# 		npmiDict[w] = {x: None for x in range(nTopics)}
# 		for x in range(nTopics):
# 			p = wordCondProbs[w][x]
# 			npmi = log2(p/pW)/(-log2(pT*p)) if p>0 else -1
# 			npmiDict[w][x] = max(npmi,0)

# 	return npmiDict



# def getNormEntropy(probs):

# 	if len(probs)<2:
# 		raise ValueError('Entropy is undefined with less than 2 values')

# 	maxEntropy = log2(len(probs))
# 	entropy = 0
# 	for p in probs:
# 		entropy -= p*log2(p) if p > 0 else 0

# 	return entropy/maxEntropy

# def getSgmtVDicts(sgmt,docList):
# 	wordProbs = {}
# 	wordCondProbs = {}

# 	pT = 1/len(sgmt)

# 	for i,(start,stop,topicIndex) in enumerate(sgmt):
# 		doc = Doc.from_docs(docList[start:stop])
# 		probs = getLemmaProbs(doc)
# 		updateDict(wordCondProbs,probs,'app',i)

# 		joints = {w: p*pT for w,p in probs.items()}
# 		updateDict(wordProbs,joints,'acc')


# 	entropies = {w: getNormEntropy([(pT*p)/wordProbs[w] for p in wordCondProbs[w]]) for w in wordProbs}
# 	npmiDict = getNPMI(wordProbs,wordCondProbs,len(sgmt))

# 	vdicts = []
# 	for i,(start,stop,topicIndex) in enumerate(sgmt):
# 		vd = {w: npmiDict[w][i] for w in wordProbs}
# 		vdicts.append(vd)

# 	return vdicts

# def getVDictCosine(vDictA,vDictB):

# 	if vDictA.keys()!=vDictB.keys():
# 		a = vDictA.keys()
# 		b = vDictB.keys()
# 		missing = (a-b) | (b-a)
# 		print(missing)
# 		raise ValueError(f'Missing words')

# 	dotSum = 0
# 	aSquareSum = 0
# 	bSquareSum = 0
# 	for key in vDictA.keys():
# 		a = vDictA[key]
# 		b = vDictB[key]
# 		dotSum += a*b
# 		aSquareSum+= pow(a,2)
# 		bSquareSum+= pow(b,2)
# 	cosine = dotSum/(sqrt(aSquareSum)*sqrt(bSquareSum))

# 	return cosine

# def segmentsIntersect(segA,segB):
# 	startA,stopA,*_ = segA
# 	startB,stopB,*_ = segB

# 	return startB<stopA and startA<stopB

# def getContentSimilarity(sgmtA,sgmtB,key,aVDicts=None,bVDicts=None):
# 	if aVDicts is None:
# 		aVDicts = getSgmtVDicts(sgmtA,key)

# 	if bVDicts is None:
# 		bVDicts = getSgmtVDicts(sgmtB,key)

# 	i = 0
# 	j = 0

# 	edges = []

# 	topBotCand = None
# 	topMaxScore = -inf
# 	botTopCand = None
# 	botMaxScore = -inf

# 	while i<len(sgmtA) and j<len(sgmtB):
# 		topStart,topStop,_ = sgmtA[i]
# 		botStart,botStop,_ = sgmtB[j]

# 		if segmentsIntersect(sgmtA[i],sgmtB[j]):
# 			edgeScore = getVDictCosine(aVDicts[i],bVDicts[j])
			
# 			if edgeScore>topMaxScore:
# 				topMaxScore = edgeScore
# 				topBotCand = j
# 			if edgeScore>botMaxScore:
# 				botMaxScore = edgeScore
# 				botTopCand = i

# 		if topStop>=botStop:
# 			edges.append((j,botTopCand,botMaxScore))
# 			botTopCand = None
# 			botMaxScore = -inf
# 			j+=1
# 		if botStop>=topStop:
# 			edges.append((i,topBotCand,topMaxScore))
# 			topBotCand = None
# 			topMaxScore = -inf
# 			i+=1

# 	score = sum([e[2] for e in edges])
# 	score = score/len(edges)

# 	return score





# def getBestDelPair(goldSgmt,goldVectors,docList,headingList,key,title):

# 	alts = []
# 	for i,(startA,stopA,tA) in enumerate(goldSgmt[:-1]):
# 		if i == 0:
# 			continue
		
# 		j = i+1

# 		(startB,stopB,tB) = goldSgmt[j]

# 		mergedSeg = (startA,stopB,f'{tA}+{tB}')

# 		h = goldSgmt[:i] + [mergedSeg] + goldSgmt[j+1:]

# 		hVDicts = getSgmtVDicts(h,docList)

# 		A = goldVectors[i]
# 		B = goldVectors[j]
# 		C = hVDicts[i]

# 		leftScore = getVDictCosine(A,C)
# 		rightScore = getVDictCosine(B,C)

# 		assert leftScore>=0
# 		assert rightScore>=0

# 		score = leftScore + rightScore

# 		delHeading = headingList[tB]
		
# 		alts.append((score,delHeading))

# 	pairTuples = []
# 	for i, (h1Score,h1Miss) in enumerate(alts):
# 		for j, (h2Score,h2Miss) in enumerate(alts[i+1:]):
# 			diff = abs(h1Score-h2Score)
# 			pairScore = diff
# 			better = 'h1' if h1Score==max(h1Score,h2Score) else 'h2'
# 			pairTuples.append((pairScore,title,key,h1Miss,h2Miss,better))

# 	pairTuples.sort(key=lambda x: x[0], reverse=True)

# 	return pairTuples[0]



# def getPairs(generator, pType='del'):

# 	if pType not in {'del','transp','ins'}:
# 		raise ValueError('Invalid pType value')

# 	pairs = []

# 	for article in generator:
# 		key = article['key']
# 		docList = list(getArticleDocBin(key).get_docs(VOCAB))
# 		try:
# 			assert docList is not None
# 			assert len(docList)==len(article['content'])
# 		except AssertionError as e:
# 			logger.warning(f'Docbin error: {key}')
# 			continue
# 		sgmt = article['segmentation']
# 		title = article['title']

# 		vectors = getSgmtVDicts(sgmt,docList)
# 		if pType=='del':
# 			pairs += [getBestDelPair(sgmt,vectors,docList,article['headings'],key,title)]

# 	pairs.sort(key=lambda x:x[0],reverse=True)

# 	return pairs


# def writePairs(pairList,fpath):

# 	lines = [','.join([str(x).replace(',','_') for x in t]) for t in pairList]
# 	with open(fpath,'w') as o:
# 		o.write('\n'.join(lines))



# def run ():

# 	articleGen = articleGenerator(DATASET_FILE)
# 	delPairs = getPairs(articleGen,pType='del')

# 	writePairs(delPairs,'./data/delPairs.csv')



# if __name__ == "__main__":
#     start = time.time()
#     logging.basicConfig(level="INFO", handlers=[
#         logging.FileHandler("./Logs/pairGenerator.log", mode='w'),
#         logging.StreamHandler()
#     ])
#     logger = logging.getLogger(__name__)
#     logger.info(f"Start:{time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(start))}")
#     run()
#     end = time.time()
#     logger.info(f"Run Time: {timedelta(seconds=end-start)}")