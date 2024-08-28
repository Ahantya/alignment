import spacy 



# step 1: wrap some article into a file that reads this text
# step 2: function where it gives you a start, stop(exclusive)
# step 3: it gives/stores out embeddings thru spacy 
# step 4: compare embeddings of each word in top segment vs bottom segment and find highest cosine similarity

def getSegment(filename, start, stop):
    with open(filename, "r") as f:
        text = f.read()

    words = text.split()
    segment = words[start:stop - 1]
    segment = ' '.join(segment)
    return segment


def main():
    segment1 = getSegment("forSpacy.txt", 0, 10)
    print(segment1)

main()
