from helpers import massLen, massToSgm, segOverlap, segmentJaccard, segToSet, massToStr, massToBinStr, segIntersectRatio
from math import inf
from functools import reduce
from nltk.metrics.segmentation import windowdiff
import segeval
from typing import List

# This is our alignment-based segmentation similarity metric; it runs in O(m1+m2).
# It receives 2 segmentations of the format [x0,x2,...x_n] where each element is a positive integer that represents the size of a segment
def endsBeforeOrSameThan(segA,segB):
    (startA,stopA) = segA
    (startB,stopB) = segB

    return stopA <= stopB

def isBetterEdge(edgeA,edgeB):
    return (edgeA['intRatio'] > edgeB['intRatio']) or (edgeA['intRatio'] == edgeB['intRatio'] and edgeA['score']>edgeB['score'])

def getUndirectedEdges(directedEdges):
    undirectedEdges = []
    seenEdgeDirections = set()
    for edge in directedEdges:
        [source, dest] = edge['direction'].split('-')
        oppositeDirection = f'{dest}-{source}'

        if (edge['direction'] not in seenEdgeDirections) and (oppositeDirection not in seenEdgeDirections):
            undirectedEdges.append(edge)
            seenEdgeDirections.add(edge['direction'])

    return undirectedEdges

def alignmentIndex(s1: List[int], s2: List[int]):
    #verify segmentations can be compared

    #get total number of elements in segmentations
    if massLen(s1) != massLen(s2):
        raise ValueError(f"Segmentations have different element length. len(s1)={massLen(s1)}; len(s2)={massLen(s2)}")

    #convert segmentations of the form [2,2] into the form [(0,1),(2,3)]; each segment is represented by the range of elems inside it (inclusive)
    s1 = massToSgm(s1)
    s2 = massToSgm(s2)

    #get the number of segments in each segmentation
    n = len(s1)
    m = len(s2)

    #set up list to store directed edges from alignment
    directedEdges = []

    #set up pointers to navigate s1(i) and s2(j)
    i = j = 0

    #set up variables to keep track of the best candidate alginment for both the ith segment in s1 and the jth segment in s2
    bestIEdge = bestJEdge = {'direction': '' ,'intRatio': -1, 'score': -1}

    #iterate left to right, starting at 0,0, until all segments in s1 and s2 have been aligned
    while i<n and j<m:

        topSeg = s1[i]
        botSeg = s2[j]

        #get candidate edges aligning s1[i] to s2[j] and s2[j] to s1[i]
        nextIEdge = {'direction': f't{i}-b{j}','intRatio': segIntersectRatio(topSeg,botSeg), 'score': segmentJaccard(topSeg, botSeg)}
        nextJEdge = {'direction': f'b{j}-t{i}','intRatio': segIntersectRatio(botSeg,topSeg), 'score': segmentJaccard(botSeg, topSeg)}

        #potentially update best candidates for i and j;

        #candidates are updated only if a)the intersect ratio is higher, b) the intersect ratio is tied and the corresponding jaccard indexes are highe
        if isBetterEdge(nextIEdge,bestIEdge):
            bestIEdge = nextIEdge

        if isBetterEdge(nextJEdge, bestJEdge):
            bestJEdge = nextJEdge

        if endsBeforeOrSameThan(topSeg,botSeg): #top seg needs to align to someone
                directedEdges.append(bestIEdge)
                bestIEdge = {'direction': '' ,'intRatio': -1, 'score': -1}
                i += 1
        if endsBeforeOrSameThan(botSeg, topSeg): #bottom seg needs to aling to someone
                directedEdges.append(bestJEdge)
                bestJEdge = {'direction': '' ,'intRatio': -1, 'score': -1}
                j += 1


    assert len(directedEdges) == n+m

    #keep only undirected edges (deletes duplicates)
    undirectedEdges = getUndirectedEdges(directedEdges)


    #sum up and average edge weights
    sum = reduce(lambda x,y: x+y['score'], undirectedEdges, 0)
    avg = sum/len(undirectedEdges)
    # print([f"({e['direction']},{e['score']:.2f})" for e in directedEdges])
    # print([f"({e['direction']},{e['score']:.2f})" for e in undirectedEdges])

    return avg

# This function is not needed but you may find it useful, it showcases how our metric, windowDiff, and B, behave on interesting examples
def compareMetricBehavior():
    print("Slide")

    h1=[1,8,1]
    h1s = massToStr(h1)
    alternates = [ [9,1], [2,7,1], [3,6,1], [4,5,1], [5,4,1], [6,3,1], [7,2,1], [8,1,1]]
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    for h2 in alternates:
        h2s = massToStr(h2)
        print(h1s)
        print(h2s)
        a = alignmentIndex(h1,h2)
        b = segeval.boundary_similarity(h1,h2)
        wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
        print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")

    print("Misc-------------------------------------")

    h1=[3,3,3,3]
    h2 = [3,2,4,3]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")

    h1=[3,3,3,3]
    h2 = [3,3,1,2,3]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")

    h1=[3,3,3,3]
    h2 = [6,3,3]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")

    h1=[3,3,3,3]
    h2 = [3,2,1,1,2,3]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")



    print("Cross-Boundary Transpositions-------------------------------------")

    h1=[5,5,1,3]
    h2 = [7,3,1,3]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")

    h1=[5,5,1,3]
    h2 = [5,4,1,4]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")




    print("Constant Cost Transp-------------------------------------")

    h1=[2,2,5,5]
    h2 = [2,2,4,6]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")

    h1=[2,2,5,5]
    h2 = [3,1,5,5]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")


    print("Vanishing Transp-------------------------------------")
    h1=[8,2,1]
    h2 = [2,8,1]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)

    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")

    h1=[7,3,1]
    h2 = [3,7,1]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)

    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")


    h1=[9,5,1]
    h2 = [5,9,1]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")

    h1=[8,6,1]
    h2 = [5,9,1]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")

    h1=[8,6,1]
    h2 = [6,8,1]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")

    h1=[7,7,1]
    h2 = [6,8,1]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")

    h1=[7,7,1]
    h2 = [7,7,1]
    h1s = massToStr(h1)
    h2s = massToStr(h2)
    k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
    print(f"k: {k}")
    print(h1s)
    print(h2s)
    a = alignmentIndex(h1,h2)
    b = segeval.boundary_similarity(h1,h2)
    wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
    print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")


    
    print("Maximal Seg Test-------------------------------------")
    h1 = [1 for x in range(10)]
    for x in range(10):
        sum = 1*x
        h2 = [1 for y in range(x)] + [10-sum]
        s1 = massToStr(h1)
        s2 = massToStr(h2)
        k = max(1,round(reduce(lambda x,y: x+y, h1)/len(h1)/2))
        print(f"k: {k}")
        print(s1)
        print(s2)
        a = alignmentIndex(h1,h2)
        b = segeval.boundary_similarity(h1,h2)
        wd = 1-windowdiff(massToBinStr(h1),massToBinStr(h2),k)
        
        print(f"z: {a:0.2f}; b: {b:0.2f}; 1-wd: {wd:0.2f};")
        print()



if __name__ == "__main__":
  	compareMetricBehavior();
