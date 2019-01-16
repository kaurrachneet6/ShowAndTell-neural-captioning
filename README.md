## Show and Tell: Neural Image Caption Generator
This project aims to build a generative model using latest techniques in Computer Vision and Machine Translation to
describe an image. Previous works in the area of Image Captioning involved combining solutions of each of the
sub-problems to generate a text description from an image. Since the goal of this project is not just getting a visual
understanding of the image but also expressing the image in a language like English, advances in translation are quite
useful. In the previous works related to Machine Translation, the translation was achieved through a series of
individual tasks like translating each word separately, reordering words etc. but there is a better way of doing this using
Recurrent Neural Networks. In RNN, an input sentence is first transformed into a compact representation using an
encoder RNN and then this representation is used as an initial hidden state of a decoder RNN that outputs the translated
sentence. In this project, a single joint model is constructed to generate a target sequence of words that describes the
input image. The idea is similar to encoder-decoder RNN used in translating sentences except that a Convolutional
Neural Network is used in place of the encoder RNN. CNN, which takes image as an input, is trained for an image
classification task to generate a compact representation of the original image. This representation is then passed as an
input to a decoder RNN that generates the sentences.

* We created training and testing scripts for 2 datasets namely:
    * MSCOCO
    * Flickr30K/8K

Inside each of the corresponding folders for the datasets, we have a train.py and test.py

* Next we have evaluation scripts for the following metrics, in the folder Eval_Scripts:
    * BLEU 1
    * BLEU 4
    * CIDer
    * ROUGE_L

* We submit .pbs files namely train.pbs and trest.pbs required to run the train and test scripts respectively.

* We also submit file to generate the qualitative results of images along with captions, called Qualitative_results.py
