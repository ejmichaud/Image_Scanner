from __future__ import division
import numpy as np
import gzip
import cPickle
import random

f = gzip.open("/home/eric/Documents/MoonVu1/images/tr1.pkl.gz")
training1 = cPickle.load(f)
f.close()
f = gzip.open("/home/eric/Documents/MoonVu1/images/tr0.pkl.gz")
training0 = cPickle.load(f)
f.close()

#THE TRAINING IMAGES
training_images = np.concatenate((training1, training0))
#THE TRAINING LABLES
training_labels = np.zeros((len(training1) + len(training0),1))
for i in xrange(len(training1)):
    training_labels[i] = 1.0

f = gzip.open("/home/eric/Documents/MoonVu1/images/te1.pkl.gz")
test1 = cPickle.load(f)
f.close()
f = gzip.open("/home/eric/Documents/MoonVu1/images/te0.pkl.gz")
test0 = cPickle.load(f)
f.close()

test_images = np.concatenate((test1, test0))
test_labels = np.zeros((200,1))
for i in xrange(100):
    test_labels[i] = 1.0

def get_test_data():
    return (test_images, test_labels)

def get_batch(k):
    batch_images = np.zeros((k, 9600))
    batch_labels = np.zeros((k,1))
    for i in xrange(k):
        n = random.randint(0, len(training_images) - 1)
        batch_images[i] = training_images[n]
        batch_labels[i] = training_labels[n]
    return (batch_images, batch_labels)
