#!/usr/bin/python

import sys

'''
A sample solution for the Bioinformatics course homework submission draft.
Petr Ryšavý <petr.rysavy@fel.cvut.cz>
'''

def read_sequences(path):
    with open(path, "r") as file_handle:
        return(file_handle.read().splitlines())

print("Hello, world!")
cpg_train = read_sequences("cpg_train.txt")
null_train = read_sequences("null_train.txt")

# maybe we should build the classifier here
test_sequences = read_sequences("seqs_test.txt")

# probably we should test the sequences to get predictions ...

print(cpg_train[0])
print(null_train[0])
print(test_sequences[0])

# TODO calculate the list of predictions, here
predictions = [0,1]

# ... and store the predictions here
with open("predictions.txt","w") as file_handle:
    file_handle.writelines(str(x)+'\n' for x in predictions)

# ... and finally, we should compare the predictions to the ground truth

test_classes = [int(cl) for cl in read_sequences("classes_test.txt")]
print(test_classes[0])

#TODO fill in the evaluation metrics ... use classes and predictions lists

correct = 0
wrong = 100
accuracy = 0.0
precision = 0.0
recall = 0.0

# ... and store the accuracy, precision, and recall
with open("accuracy.txt","w") as file_handle:
    file_handle.writelines(str(x)+'\n' for x in [correct, wrong, accuracy, precision, recall])

print("Goodbye, world!")


