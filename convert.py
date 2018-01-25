from __future__ import division
from sys import argv
import PIL.Image as Image
import numpy as np
import model
import matplotlib.pyplot as plt

WIDTH = 1700
HEIGHT = 900

img = Image.open('../images/training1/925.jpg')
img = img.resize((WIDTH,HEIGHT), Image.ANTIALIAS)
i = np.array(img.getdata())
i = np.mean(i, axis=1)
i = i.reshape((HEIGHT, WIDTH))

def detect(metaframe):
	metaframe = (metaframe - np.mean(metaframe)) / np.std(metaframe)
	counter = np.zeros(metaframe.shape)
	plt.imshow(metaframe)
	plt.show()
	total_count = 0
	avg = 0
	canvas = np.zeros(metaframe.shape)
	for n in xrange(0, HEIGHT-80, 30):
		for k in xrange(0, WIDTH-120, 30):
			probability = model.run(metaframe[n:n+80,k:k+120])
			canvas[n:n+80, k:k+120] += probability
			counter[n:n+80, k:k+120] += 1
			avg += probability
			total_count += 1
	canvas = canvas - ((avg / total_count) * counter)
	return canvas

plt.imshow(detect(i), interpolation='gaussian', cmap='seismic')
plt.show()
