ReadME for alignmentIndex Method


Alignment-Based Segmentation Similarity Metric

Introduction:

The Alignment-Based Segmentation Similarity Metric is a method used to compare two segmentations, which are ways of dividing a sequence into segments of different lengths. This metric measures how similar the two segmentations are and provides a similarity score, where higher scores indicate greater similarity between the segmentations.

Input:

The method takes two segmentations as input, each represented as a list of positive integers. These integers represent the size of each segment. For example, the segmentation [3, 3, 1, 5] means the first segment has 3 elements, the second segment has 3 elements, the third segment has 1 element, and the fourth element has 5 elements.

Output:

The output of the method is a similarity score that quantifies the degree of similarity between the two segmentations.


How it works:

The method first verifies if the two segmentations can be compared. To be compared, they must have the same total number of elements.

It converts the segmentations into a different representation where each segment is represented by a range of elements (inclusive). For example, [3, 3, 1, 5] is converted to [(0, 2), (3, 5), (6, 6), (7, 11)]

The first segment has 3 elements, starting from index 0 and ending at index 2 (inclusive). So, it becomes (0, 2).

The second segment also has 3 elements, starting from index 3 and ending at index 5 (inclusive). So, it becomes (3, 5).

The third segment has 1 element, starting from index 6 and ending at index 6 (inclusive). So, it becomes (6, 6).

The fourth segment has 5 elements, starting from index 7 and ending at index 11 (inclusive). So, it becomes (7, 11).

Converted Segmentation: [(0, 2), (3, 5), (6, 6), (7, 11)]

For each pair of overlapping segments, it calculates two intersect ratios, one for each segmentation. The intersect ratio measures how much the segments overlap with each other.

Then, it keeps track of the best candidate segments for alignment for each segment in both segmentations based on the intersect ratio and Jaccard index. Jaccard index measures the similarity between two sets.

Once a segment in one segmentation reaches further right than the corresponding segment in the other segmentation, it means all candidates for alignment have been considered. The method then saves the weighted edges representing the alignment with the best intersect ratio. Then, it moves the pointers/variables to the next segments to be aligned. 

This process continues until all segments have been aligned.

Finally, the method calculates the average edge weight (Jaccard index) by dividing the sum of the weights by the number of undirected edges in the undirectedEdges set.

A higher score means that the two segments are aligned better 