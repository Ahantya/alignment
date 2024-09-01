import spacy
import numpy as np
from spacy.tokens import DocBin, Doc    


# step 1: wrap some article into a file that reads this text
# step 2: function where it gives you a start, stop(exclusive)
# step 3: it gives/stores out embeddings thru spacy 
# step 4: compare embeddings of each word in top segment vs bottom segment and find highest cosine similarity
# actually average cosine similarities


nlp = spacy.load("en_core_web_md")

            # if similarity > highestSimilarity:
            #     highestSimilarity = similarity
            #     mostSimilarWords = (word1, word2)


def getSpanText(filename, start, stop):
    text = loadDocBin(filename)
    
    for doc in text:
        segment = doc.text.split()
        segment = segment[start:stop]
        segment = " ".join(segment)


        vectors = [token.vector for token in doc]
        segmentVectors = vectors[start:stop]

        print("Segment Text:", segment)
        print("Segment Vectors:", segmentVectors)

        return segment, segmentVectors
        




def saveDocBin(filename, filepath):
    with open(filename, "r") as f:
        text = f.read() 


    doc = nlp(text)
    doc_bin = DocBin()
    doc_bin.add(doc)

    doc_bin.to_disk(filepath) # only needed to be used once

def loadDocBin(filepath):
    doc_bin = DocBin().from_disk(filepath)

    docs = list(doc_bin.get_docs(nlp.vocab))

    return docs





# def embeddingStore(segment1, segment2):
#     nlp = spacy.load('en_core_web_md')
#     doc1 = nlp(segment1) # scale this and change this
#     # multiple paragraphs
#     doc2 = nlp(segment2) #nlp.pipe instead of this and then store the txt storing spacy docs to disk
    
#     embeddingsList1 = {}
#     embeddingsList2 = {}

#     for token in doc1:
#         embeddingsList1[token.text] = token.vector

#     for token in doc2:
#         embeddingsList2[token.text] = token.vector
    
    #return embeddingsList1, embeddingsList2

# def cosineSimilarity(embedding1, embedding2):
#     dot_product = np.dot(embedding1, embedding2)
#     norm_embedding1 = np.linalg.norm(embedding1)
#     norm_embedding2 = np.linalg.norm(embedding2)
#     return dot_product / (norm_embedding1 * norm_embedding2)

# use the lemma 

# def compareSegments(segment1, segment2): # parameters: two ranges, process embeddings from docs and get what i need
#     embeddingsList1, embeddingsList2 = embeddingStore(segment1, segment2)
#     totalSimilarity = 0
#     pairCount = 0

#     for embedding1 in embeddingsList1.values():
#         for embedding2 in embeddingsList2.values():
#             similarity = cosineSimilarity(embedding1, embedding2)
#             totalSimilarity += similarity
#             pairCount += 1
#     if pairCount == 0:
#         return 0

#     averageSimilarity = totalSimilarity / pairCount 
#     return averageSimilarity

def main():
    # segment1 = getSegment("spaCyWork/forSpacy.txt", 0, 10)
    # segment2 = getSegment("spaCyWork/forSpacy.txt", 200, 210)
    # similarity = compareSegments(segment1, segment2)
    # print(f"Cosine Average Similarity: {similarity:.4f}")
    filename = "spaCyWork/forSpacy.txt"
    docbin_filepath = "full_text.spacy"
    #saveDocBin(filename, docbin_filepath)


    getSpanText(docbin_filepath, 0, 10)



main()
