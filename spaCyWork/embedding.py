import spacy
import numpy as np


# step 1: wrap some article into a file that reads this text
# step 2: function where it gives you a start, stop(exclusive)
# step 3: it gives/stores out embeddings thru spacy 
# step 4: compare embeddings of each word in top segment vs bottom segment and find highest cosine similarity
# actually average cosine similarities




            # if similarity > highestSimilarity:
            #     highestSimilarity = similarity
            #     mostSimilarWords = (word1, word2)



import spacy
import numpy as np

def getSegment(filename, start, stop):
    with open(filename, "r") as f:
        text = f.read()

    words = text.split()
    segment = words[start:stop]
    segment = ' '.join(segment)
    return segment

def embeddingStore(segment1, segment2):
    nlp = spacy.load('en_core_web_md')
    doc1 = nlp(segment1)
    doc2 = nlp(segment2)
    
    embeddingsList1 = {}
    embeddingsList2 = {}

    for token in doc1:
        embeddingsList1[token.text] = token.vector

    for token in doc2:
        embeddingsList2[token.text] = token.vector
    
    return embeddingsList1, embeddingsList2

def cosineSimilarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    return dot_product / (norm_embedding1 * norm_embedding2)

def compareSegments(segment1, segment2):
    embeddingsList1, embeddingsList2 = embeddingStore(segment1, segment2)
    totalSimilarity = 0
    pairCount = 0

    for embedding1 in embeddingsList1.values():
        for embedding2 in embeddingsList2.values():
            similarity = cosineSimilarity(embedding1, embedding2)
            totalSimilarity += similarity
            pairCount += 1
    if pairCount == 0:
        return 0

    averageSimilarity = totalSimilarity / pairCount 
    return averageSimilarity

def main():
    segment1 = getSegment("forSpacy.txt", 0, 10)
    segment2 = getSegment("forSpacy.txt", 200, 210)
    similarity = compareSegments(segment1, segment2)
    print(f"Cosine Average Similarity: {similarity:.4f}")

main()
