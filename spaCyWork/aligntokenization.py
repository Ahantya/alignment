import spacy
from spacy.training import Alignment

def alignTokens(combinedDoc, otherTokens):
   
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(combinedDoc)

    spacy_tokens = [token.text for token in doc]
    
    align = Alignment.from_strings(otherTokens, spacy_tokens)
    
    print(f"a -> b, lengths: {align.x2y.lengths}")  # array([1, 1, 1, 1, 1, 1, 1, 1])
    print(f"a -> b, mapping: {align.x2y.data}")  # array([0, 1, 2, 3, 4, 4, 5, 6])
    print(f"b -> a, lengths: {align.y2x.lengths}")  # array([1, 1, 1, 1, 2, 1, 1])
    print(f"b -> a, mappings: {align.y2x.data}")  # array([0, 1, 2, 3, 4, 5, 6, 7])


combinedDoc = "I listened to Obama's podcasts."
otherTokens = ["i", "listened", "to", "obama", "'", "s", "podcasts", "."]
alignTokens(combinedDoc, otherTokens)
