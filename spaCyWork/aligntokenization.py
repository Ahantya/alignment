import spacy
from spacy.training import Alignment


nlp = spacy.load("en_core_web_md")

text = "I like driving Joe's car, especially at night."

doc = nlp(text)

# given this from spacing
iStart = 0
iStop = 5  


other_tokens = text.split(" ")

otherWord = other_tokens[iStart:iStop] #this gets the word from the index istart to iStop


spacy_tokens = [token.text for token in doc]
align = Alignment.from_strings(other_tokens, spacy_tokens)

# print(other_tokens)
# print(spacy_tokens)


mappedSpacyIndices = []

for i in range(iStart, iStop + 1):
    spacyIndices = align.x2y[i]
    mappedSpacyIndices.extend(spacyIndices)

print("(" + (str(mappedSpacyIndices[0]) + ", " + str(mappedSpacyIndices[-1]) + ")"))

# print(f"a -> b, lengths: {align.x2y.lengths}")  
# print(f"a -> b, mapping: {align.x2y.data}")   #***
# print(f"b -> a, lengths: {align.y2x.lengths}")  


print(f"b -> a, mappings: {align.y2x.data}")   #***


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
