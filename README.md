# Image_Scanner
A system for scanning images for objects using a multi-layer convolutional neural network.

The goal here is to explore the frame-sliding method of object detection.

To start, I trained a CNN on a dataset split into two classes. Images either do or do not
contain cars. The network achieves about 90% accuracy on the training data.

To identify regions of an image that contain cars, I slide a "frame" over the image,
sampling regions, and running the network on these regions. This generates a distribution
over the dimensions of the image, where higher numbers correspond with a higher probability
that a car is in that region.

Here is a test...

Raw Image:

![alt text](results/car.jpg?raw=true "Car Raw")

Distribution:

![alt text](results/car1_map.jpg?raw=true "Car Scanned")

