# Visual Question Answering

This is a relatively simple approach to the VQA challenge, implemented for
Princeton's COS 529 course, where the aim is to answer unseen questions about
unseen images. The dataset can be found at:

https://visualqa.org/download.html

The model was trained and tested on the balanced real images.

## Approach

The model uses an LSTM, based on spaCy word embeddings, for feature extraction from questions. We use pre-extracted image features from a ResNet 101. The two are combined by element-wise multiplication, and then fed into two fully connected layers.

### Additional Approaches

There are also options for the user to try the model with word embeddings only, and to use the question types for inference.

## Requirements

* Tensorflow < 2
* Keras
* spaCy
