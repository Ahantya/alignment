import spacy
from spacy.training import Alignment


nlp = spacy.load("en_core_web_md")

text = "I sent tom's girlfriend sally's clothes"

doc = nlp(text)

# given this from spacing
iStart = 2
iStop = 5  # excludes this last index


space_tokens = text.split(" ")
#doc.text.split(" ")

#otherWord = other_tokens[iStart:iStop] #this gets the word from the index istart to iStop


spacy_tokens = [token.text for token in doc]
print(space_tokens)
print(spacy_tokens)
align = Alignment.from_strings(space_tokens, spacy_tokens)

spacyStart = align.x2y[iStart][0]
spacyEnd = align.x2y[iStop][-1]

# function spaceRangeToSpacyRange(start, stop, currentDoc (represents text))
    #doc.text.split(" ") for space tokens
    #spacy tokens as it is
    # align = Alignment.from_strings(space_tokens, spacy_tokens)
    #spacyStart = align.x2y[iStart][0]
    #spacyEnd = align.x2y[iStop][-1] 
    #return (spacyStart, spacyEnd) tuple 

spacyRange = (spacyStart, spacyEnd)
print(spacy_tokens[spacyStart:spacyEnd])

#mappedSpacyIndices = []

# for i in range(iStart, iStop + 1):
#     spacyIndices = align.x2y[i]
#     print(spacyIndices, i)
#     mappedSpacyIndices.extend(spacyIndices) # so we dont have lists within lists

#print("(" + (str(mappedSpacyIndices[0]) + ", " + str(mappedSpacyIndices[-1]) + ")"))

# range for OUTPUT

# print(f"a -> b, lengths: {align.x2y.lengths}")  
# print(f"a -> b, mapping: {align.x2y.data}")   #***
# print(f"b -> a, lengths: {align.y2x.lengths}")  


#print(f"b -> a, mappings: {align.y2x.data}")   #***


# i need this once i get spacy's based
# oStart = 0
# oStop = 7

# some type of loop around the mapping array?


# a to b mapping should be look like: [0 1 2 3 5 7 8 9]

#other ['I', 'like', 'driving', "Joe's", 'car,', 'especially', 'at', 'night.']
#spacy ['I', 'like', 'driving', 'Joe', "'s ", 'car', ',', 'especially', 'at', 'night', '.']


# a -> b, lengths: [1 1 1 2 2 1 1 2]
# a -> b, mapping: [ 0  1  2  3  4  5  6  7  8  9 10]
# b -> a, lengths: [1 1 1 1 1 1 1 1 1 1 1]
# b -> a, mappings: [0 1 2 3 3 4 4 5 6 7 7] # try this


# other_tokens = ["i", "sent", "tom's",   "girlfiend", "sally", "'s", "clothes"]
# spacy_tokens = ["i", "sent", "tom","'s","girlfiend", "sally's",     "clothes"]
# align = Alignment.from_strings(other_tokens, spacy_tokens)
# print(f"a -> b, lengths: {align.x2y.lengths}")  # array([1, 1, 1, 1, 1, 1, 1, 1])
# print(f"a -> b, mapping: {align.x2y.data}")  # array([0, 1, 2, 3, 4, 4, 5, 6]) : two tokens both refer to "'s"
# print(f"b -> a, lengths: {align.y2x.lengths}")  # array([1, 1, 1, 1, 2, 1, 1])   : the token "'s" refers to two tokens
# print(f"b -> a, mappings: {align.y2x.data}")  # array([0, 1, 2, 3, 4, 5, 6, 7])


# #print(align.x2y[2])

