from __future__ import division
import PIL
from PIL import Image
import numpy as np
import sys
from os import listdir
import matplotlib.pyplot as plt
import cPickle
import gzip

DIRECTORY = "/home/eric/Documents/MoonVu1/images/test1"
FILE_NAMES = listdir(DIRECTORY)
data = np.zeros((len(FILE_NAMES), 80*120))

counter = 0
for file_name in FILE_NAMES:
	#open and resize the image
	img = Image.open(DIRECTORY + "/" + file_name)
	img = img.resize((120, 80), Image.ANTIALIAS)
	#transition to numpy array
	img_array = np.array(img.getdata())
	#converts RGB to MONO
	if len(img_array.shape) > 1:
		img_array = np.mean(img_array, axis=1)
	#normalize
	img_array = (img_array - np.mean(img_array)) / np.std(img_array)
	#check for proper format
	if img_array.shape != (9600,):
		print "size error on {}".format(file_name)
		break
	data[counter] = img_array
	counter += 1
	if counter % 10 == 0:
		print "{} / {} processed".format(counter, len(FILE_NAMES))

print "SAVING..."
f = gzip.open("/home/eric/Documents/MoonVu1/images/" + "te1.pkl.gz", "w")
cPickle.dump(data, f)
f.close()
print "DONE!"
