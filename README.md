The blog post (https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/) implements an image captioning neural network in keras.

This repository aims to implement an entire image captioning model in tensorflow. Except for the feature extraction stage, everything else is written using the tensorflow (and numpy) library only.

Features are extracted using a pretrained VGG16 neural net. While Keras has these pretrained models in its library, tensorflow does not. Implementing VGG16 is another project in itself. So for now, Keras is used for this stage. But I'll try to implement the VGG16 architecture in another project in the future.

