# naivebayes.py
"""Perform document classification using a Naive Bayes model."""

import argparse
import os
import pdb
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

ROOT = 'data'  # change to path where data is stored

parser = argparse.ArgumentParser(description="Use a Naive Bayes model to classify text documents.")
parser.add_argument('-x', '--training_data',
                    help='path to training data file, defaults to ROOT/trainingdata.txt',
                    default=os.path.join(ROOT, 'trainingdata.txt'))
parser.add_argument('-y', '--training_labels',
                    help='path to training labels file, defaults to ROOT/traininglabels.txt',
                    default=os.path.join(ROOT, 'traininglabels.txt'))
parser.add_argument('-xt', '--testing_data',
                    help='path to testing data file, defaults to ROOT/testingdata.txt',
                    default=os.path.join(ROOT, 'testingdata.txt'))
parser.add_argument('-yt', '--testing_labels',
                    help='path to testing labels file, defaults to ROOT/testinglabels.txt',
                    default=os.path.join(ROOT, 'testinglabels.txt'))
parser.add_argument('-n', '--newsgroups',
                    help='path to newsgroups file, defaults to ROOT/newsgroups.txt',
                    default=os.path.join(ROOT, 'newsgroups.txt'))
parser.add_argument('-v', '--vocabulary',
                    help='path to vocabulary file, defaults to ROOT/vocabulary.txt',
                    default=os.path.join(ROOT, 'vocabulary.txt'))


def main(args):
    print("Document Classification using Naïve Bayes Classifiers")
    print("=======================")
    print("PRE-PROCESSING")
    print("=======================")

    # Parse input arguments
    training_data_path = os.path.expanduser(args.training_data)
    training_labels_path = os.path.expanduser(args.training_labels)
    testing_data_path = os.path.expanduser(args.testing_data)
    testing_labels_path = os.path.expanduser(args.testing_labels)
    newsgroups_path = os.path.expanduser(args.newsgroups)
    vocabulary_path = os.path.expanduser(args.vocabulary)

    # Load data from relevant files
    # ***MODIFY CODE HERE***
    trainingData = np.loadtxt(training_data_path, dtype=int)
    trainingLabels = np.loadtxt(training_labels_path, dtype=int)
    testData = np.loadtxt(testing_data_path, dtype=int)
    testLabels = np.loadtxt(testing_labels_path, dtype=int)
    newsGroupsTxt = np.loadtxt(newsgroups_path, dtype=str)
    vocab = np.loadtxt(vocabulary_path, dtype=str)
    print("Loading training data...")
    xtrain = trainingData
    print("Loading training labels...")
    ytrain = trainingLabels
    print("Loading testing data...")
    xtest = testData
    print("Loading testing labels...")
    ytest = testLabels
    print("Loading newsgroups...")
    newsgroups = newsGroupsTxt
    print("Loading vocabulary...")
    vocabulary = vocab

    # Change 1-indexing to 0-indexing for labels, docID, wordID
    # ***MODIFY CODE HERE***
    for i in range(len(ytrain)):
        ytrain[i] -= 1
    for i in range(len(ytest)):
        ytest[i] -= 1
    for i in range(len(xtrain)):
        xtrain[i][0] -= 1
        xtrain[i][1] -= 1
    for i in range(len(xtest)):
        xtest[i][0] -= 1
        xtest[i][1] -= 1

    # Extract useful parameters
    num_training_documents = len(ytrain)
    num_testing_documents = len(ytest)
    num_words = len(vocabulary)
    num_newsgroups = len(newsgroups)

    print("\n=======================")
    print("TRAINING")
    print("=======================")

    # Estimate the prior probabilities
    print("Estimating prior probabilities via MLE...")
    # ***MODIFY CODE HERE***
    currDoc = 0
    docCounts = np.zeros(20)
    for dataLine in xtrain:
        if dataLine[0] != currDoc:
            docCounts[ytrain[dataLine[0]]] += 1
            currDoc = dataLine[0]
        dataLine[0] = ytrain[dataLine[0]]
    priors =  docCounts / num_training_documents

    # Estimate the class conditional probabilities
    print("Estimating class conditional probabilities via MAP...")
    # ***MODIFY CODE HERE***
    class_conditionals = np.zeros((num_newsgroups, num_words))
    for dataPoint in xtrain:
        class_conditionals[dataPoint[0], dataPoint[1]] += dataPoint[2]
    virtualData = 1 / num_words
    for i in range(20):
        num_words_in_doctype = np.sum(class_conditionals[i, :])
        class_conditionals[i, :] += virtualData
        class_conditionals[i, :] /= num_words_in_doctype

    print("\n=======================")
    print("TESTING")
    print("=======================")

    # Test the Naive Bayes classifier
    print("Applying natural log to prevent underflow...")
    log_priors = np.log(priors)
    log_class_conditionals = np.log(class_conditionals)

    print("Counting words in each document...")
    # ***MODIFY CODE HERE***
    counts = np.zeros((num_testing_documents, num_words))
    for dataPoint in xtest:
        counts[dataPoint[0], dataPoint[1]] += dataPoint[2]
    # pdb.set_trace()

    print("Computing posterior probabilities...")
    log_posteriors = np.zeros(num_testing_documents * 20)
    for i in range(num_testing_documents):
        posterior = np.matmul(log_class_conditionals[:, :], counts[i, :]) + log_priors
        log_posteriors[i * 20:i * 20 + 20] += posterior

    print("Assigning predictions via argmax...")
    # ***MODIFY CODE HERE***
    # pred = np.zeros(num_testing_documents)
    pred = [0] * num_testing_documents
    for i in range(num_testing_documents):
        pred[i] = np.argmax(log_posteriors[i * 20:i * 20 + 20]) % 20

    print("\n=======================")
    print("PERFORMANCE METRICS")
    print("=======================")

    # Compute performance metrics
    # ***MODIFY CODE HERE***
    accuracy = (np.bincount(np.equal(pred, ytest))[1] / num_testing_documents) * 100
    # pdb.set_trace()
    print("Accuracy: {0}".format(accuracy))
    cm = confusion_matrix(ytest, pred)
    print("Confusion matrix:")
    print(cm)

    # pdb.set_trace()  # uncomment for debugging, if needed


if __name__ == '__main__':
    main(parser.parse_args())
