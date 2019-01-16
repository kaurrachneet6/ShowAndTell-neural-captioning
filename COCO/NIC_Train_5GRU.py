'''
CS 598 Final Project: Show and Tell
Neural Image Captioning 
Team Members: Avinash, Ankit Kumar, Daksh Saraf, Rachneet Kaur

Script to train the encoding_architecture and decoding_architecture model
'''

#Importing libraries needed
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import h5py
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
from pycocotools.coco import COCO #For MSCOCO dataset
from collections import Counter
import nltk #For evaluation metrics 
#nltk.download('punkt')
import pickle
from itertools import takewhile
from PIL import Image
import torch.utils.data as data
import os
from datetime import datetime
from torchvision.datasets import CocoCaptions #For MSCOCO dataset captions 
import time

'''For distribured learning on BlueWaters'''
#import torch.distributed as dist
#import subprocess
#from mpi4py import MPI

#Defining the dictionary for the needed paths and parameters 
parameter_dict = {'batch_size':32, 
                #Larger Batch sizes give out of memory errors on Blue Waters, hence choose 32  
              'shuffle':True,
              'num_workers':32,
              'data_dir': '/projects/training/bauh/COCO', 
                #MSCOCO dataset 
              'output_dir': '/u/training/tra402/scratch/Project/Normal_5GRU/Output', 
                #Storing the model checkpoints, to retrain if needed
              'train_ann_path': 'annotations/captions_train2014.json',
               #Path for the captions for the training dataset in MSCOCO
              'test_ann_path': 'annotations/captions_val2014.json',
               #Path for the captions for the validation dataset in MSCOCO
              'vocabulary_path':'vocab.pkl',
               #Vocabulary file 
              'train_img_dir':'train2014',
                #Folder for the train dataset images in the MSCOCO dataset
              'test_img_dir': 'val2014',
               #Folder for the validation images in the MSCOCO dataset
              'vocab_threshold':5}
               #Consider a word in the vocabulary only if it appears at least 5 times 

#Class to build the vocabulary and assign start and end token whenever necessary
class DatasetVocabulary(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.index = 0

    def adding_new_word(self, word): 
        #Adding a new word to the vocabulary, if it already doesn't exist
        if not word in self.word_to_index:
            self.word_to_index[word] = self.index
            self.index_to_word[self.index] = word
            self.index += 1

    def __call__(self, word):
        if not word in self.word_to_index:
            #If word does not exist in vocabulary, then return unknown token
            return self.word_to_index['<unk>']
        return self.word_to_index[word]

    def __len__(self):
        #Returns the length of the vocabulary 
        return len(self.word_to_index)

    def start_token(self):
        #Returns the start token 
        return '<start>'

    def end_token(self):
        #Returns the end token 
        return '<end>'

#Function for creating the vocabulary for the MSCOCO dataset
def creating_vocabulary(json_file):
    coco_json = COCO(json_file)
    vocab_word_ids = coco_json.anns.keys()
    for index, word_id in enumerate(vocab_word_ids):
        #Converting all the words to lower case and tokenizing them
        captions_tokens = nltk.tokenize.word_tokenize(str(coco_json.anns[word_id]['caption']).lower())
        Counter().update(captions_tokens)
    #We only consider the words which appear more than a particular threshold 
    vocabulary_words = []
    for vocab_word, vocab_word_count in Counter().items():
        if vocab_word_count >=parameter_dict['vocab_threshold']:
            vocabulary_words.append(vocab_word)
    vocabulary_dataset = DatasetVocabulary()    
    vocabulary_dataset.adding_new_word('<pad>')
    vocabulary_dataset.adding_new_word('<start>')
    vocabulary_dataset.adding_new_word('<end>')
    vocabulary_dataset.adding_new_word('<unk>')

    for index, vocab_word in enumerate(vocabulary_words):
        vocabulary_dataset.adding_new_word(vocab_word)
    return vocabulary_dataset

#Defining the path for the vocabulary
vocabulary_path = os.path.join(parameter_dict['output_dir'], parameter_dict['vocabulary_path'])

#Loading the vocabulary from the vocabulary file
def get_vocab(vocabulary_path):
    if(os.path.isfile(vocabulary_path)):
        #If the file is already craeted and exists
        with open(vocabulary_path, 'rb') as f:
            vocabulary = pickle.load(f)
            print('Vocabulary is loaded from the pickle file')
    else:
        #Else creating the vocabulary file 
        vocabulary = creating_vocabulary(json_file=os.path.join(parameter_dict['data_dir'], parameter_dict['train_ann_path']))
        with open(vocabulary_path, 'wb') as f:
            pickle.dump(vocabulary, f)
    return vocabulary

#Data loader for the MSCOCO dataset
class MSCOCO(data.Dataset): 
    def __init__(self, annotations, data_path, vocabulary, augmentation_transforms=None):
        #Specifying the path for the datasets 
        self.data_path = data_path
        #Defining the vocabulary for the dataset
        self.vocabulary = vocabulary
        #Specifying the data augmentations on the dataset
        self.augmentation_transforms = augmentation_transforms
        #Creating a list of the annotation IDs for MSCOCO dataset
        self.annotation_ids = list(COCO(annotations).anns.keys())

    def __getitem__(self, image_index):
        '''
        Function to retrieve an image and it's corresponding caption for the dataset
        '''
        #Retrieving the annotation index corresponding to the image index
        annotation_index = self.annotation_ids[image_index]
        #Retrieving the caption corresponding to the image index
        image_caption = COCO(annotations).anns[annotation_index]['caption']
        #Retrieving the image index corresponding to the annotation index from Image index
        image_index = COCO(annotations).anns[annotation_index]['image_id']
        #Recording the path of the image 
        image_path = COCO(annotations).loadImgs(image_index)[0]['file_name']
        #Applying the data augmentation transformations on the images 
        image = Image.open(os.path.join(self.data_path, image_path)).convert('RGB')
        if self.augmentation_transforms is not None:
            image = self.augmentation_transforms(image)

        #Tokenizing the captions for the image after converting them to lower case
        tokens_caption = nltk.tokenize.word_tokenize(str(image_caption).lower())
        image_target_caption = []
        #Starting the caption with <start>
        image_target_caption.append(vocabulary('<start>'))
        image_target_caption.extend([vocabulary(token) for token in tokens_caption])
        #Ending the caption with <end>
        image_target_caption.append(vocabulary('<end>'))
        #Converting the tokenized caption to a tensor for computations 
        image_target_caption = torch.Tensor(image_target_caption)
        #Returning the image and it's corresponding real caption for the index given
        return image, image_target_caption

    def __len__(self):
        '''
        Function returning the length of the dataset
        '''
        return len(self.annotation_ids)

def create_batch(data):
    ''' 
    Function to create batches from images and it's corresponding real captions
    '''
    #Sorting 
    data.sort(key=lambda x: len(x[1]), reverse=True)
    #Retrieving the images and their corresponding captions 
    dataset_images, dataset_captions = zip(*data)
    #Stacking the images together 
    dataset_images = torch.stack(dataset_images, 0)
    #Writing the lengths of the image captions to a list
    caption_lengths = []
    for caption in dataset_captions:
        caption_lengths.append(len(caption))
    target_captions = torch.zeros(len(dataset_captions), max(caption_lengths)).long()
    for index, image_caption in enumerate(dataset_captions):
        caption_end = caption_lengths[index]
        #Computing the length of the particular caption for the index 
        target_captions[index, :caption_end] = image_caption[:caption_end]
    #Returns the images, captions and lengths of captions according to the batches 
    return dataset_images, target_captions, caption_lengths

def get_data_transforms():
    '''
    Function to apply data Transformations on the dataset 
    '''
    data_trans = transforms.Compose([transforms.Resize((224, 224)), 
                                     #Resize to 224 because we are using pretrained Imagenet
                                     transforms.RandomHorizontalFlip(),
                                     #Random horizontal flipping of images 
                                     transforms.RandomVerticalFlip(),
                                     #Random vertical flipping of images 
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406),
                                                          (0.229, 0.224, 0.225))])
                                     #Normalizing the images    
    #Returning the  transformed images                                 
    return data_trans

def get_data_loader(annotations_path, data_path, vocabulary, data_transforms, parameter_dict):
    ''' Function to load the required dataset in batches '''
    dataset = MSCOCO(annotations_path, data_path, vocabulary, data_transforms)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=parameter_dict['batch_size'],
                                              shuffle=parameter_dict['shuffle'],
                                              num_workers=parameter_dict['num_workers'],
                                              collate_fn=create_batch)
    return data_loader

class Resnet(nn.Module):
    ''' Class for defining the CNN architecture implemetation'''
    def __init__(self):
        super(Resnet, self).__init__()
        self.resultant_features  = 80
        #Loading the pretrained Resnet model on ImageNet dataset
        #We tried Resnet50/101/152 as architectures 
        resnet_model = models.resnet101(pretrained=True)
        self.model = nn.Sequential(*list(resnet_model.children())[:-1])
        #Training only the last 2 layers for the Resnet model i.e. linear and batchnorm layer
        self.linear_secondlast_layer = nn.Linear(resnet_model.fc.in_features, self.resultant_features)
        #Last layer is the 1D batch norm layer
        self.last_layer = nn.BatchNorm1d(self.resultant_features, momentum=0.01)
        #Initializing the weights using normal distribution 
        self.linear_secondlast_layer.weight.data.normal_(0,0.05)
        self.linear_secondlast_layer.bias.data.fill_(0)

    def forward(self, input_x):
        ''' Defining the forward pass of the CNN architecture model'''
        input_x = self.model(input_x)
        #Converting to a pytorch variable 
        input_x = Variable(input_x.data)
        #Flattening the output of the CNN model
        input_x = input_x.view(input_x.size(0), -1)
        #Applying the linear layer 
        input_x = self.linear_secondlast_layer(input_x)
        return input_x

def create_caption_word_format(tokenized_version, dataset_vocabulary):
    ''' Function to convert the tokenized version of sentence 
    to a sentence with words from the vocabulary'''
    
    #Defining the start token
    start_word = [dataset_vocabulary.word_to_index[word] for word in [dataset_vocabulary.start_token()]]
    #Defining the end token
    end_word = lambda index: dataset_vocabulary.index_to_word[index] != dataset_vocabulary.end_token()
    #Creating the sentence in list format from the tokenized version of the result 
    caption_word_format_list = []
    for index in takewhile(end_word, tokenized_version)
        if index not in start_word:
            caption_word_format_list.append(dataset_vocabulary.index_to_word[index]) 
    #Returns the sentence with words from the vocabulary
    return ' '.join(caption_word_format_list)

class RNN(torch.nn.Module):
    ''' Class to define the RNN implementation '''
    def __init__(self, embedding_length, hidden_units, vocabulary_size, layer_count):
        super(RNN, self).__init__()
        #Defining the word embeddings based on the embedding length = 512 and vocabulary size 
        self.embeddings = nn.Embedding(vocabulary_size, embedding_length)
        #Defining the hidden unit to be LSTM unit or GRU unit with hidden_units no. of units
        self.unit = nn.GRU(embedding_length, hidden_units, layer_count, batch_first=True)
        #Defining the last linear layer converting to the vocabulary_size
        self.linear = nn.Linear(hidden_units, vocabulary_size)

    def forward(self, CNN_feature, image_caption, caption_size):
        ''' Defining the forward pass of the RNN architecture model'''
        #Creating the embeddings for the image captions 
        caption_embedding = self.embeddings(image_caption)
        torch_raw_embeddings = torch.cat((CNN_feature.unsqueeze(1), caption_embedding), 1)
        torch_packed_embeddings = nn.utils.rnn.pack_padded_sequence(torch_raw_embeddings, caption_size, batch_first=True)
        torch_packed_embeddings_unit= self.unit(torch_packed_embeddings)[0]
        tokenized_predicted_sentence = self.linear(torch_packed_embeddings_unit[0])
        #Return the predicted sentence in the tokenized version which need to be converted to words 
        return tokenized_predicted_sentence

    def sentence_index(self, CNN_feature):
        #Defining the maximum caption length 
        caption_max_size = 25 
        #Defining the RNN hidden state to be None in the beginning 
        RNN_hidden_state = None
        #Defining the input for the RNN based on the CNN features
        RNN_data = CNN_feature.unsqueeze(1)
        #To return the predicted sentence tokenized version 
        predicted_sentence_index = []
        for i in range(caption_max_size):
            #Predicting each next hidden state and word based on the RNN model 
            next_state, RNN_hidden_state = self.unit(RNN_data, RNN_hidden_state)
            #Linear layer 
            result_state = self.linear(next_state.squeeze(1))
            #Predicted word based on the model
            predicted_tokenized_word = result_state.max(1)[1]
            #Appending the index for the word
            predicted_sentence_index.append(predicted_tokenized_word)
            #Applying the embeddings to the predicted word in tokenized version 
            RNN_data = self.embeddings(predicted_tokenized_word)
            RNN_data = RNN_data.unsqueeze(1)
        #Stacking all the predicted tokenized words 
        predicted_sentence_index = torch.stack(predicted_sentence_index, 1).squeeze()
        #Returning the tokenized version of the predicted sentence
        return predicted_sentence_index

def create_checkpoint(encoding_architecture, decoding_architecture, optimizer, epoch, step, losses_train, parameter_dict):
    ''' Function to create a checkpoint for the trained models and their corresponding 
    evaluated metrics '''
    #Saving the .ckpt model file 
    model_file = 'model_'+str(epoch+1)+'.ckpt'
    #Saving the .ckpt file for the metrics of the trained model
    metrics_file = 'model_'+str(epoch+1)+'_metrics.ckpt'
    #Saving the dictionary corresponding to the trained model inorder to retrain again 
    torch.save({'encoder_state_dict': encoding_architecture.state_dict(),
                'decoder_state_dict': decoding_architecture.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'epoch':epoch,
                'step':step}, 
               os.path.join(parameter_dict['output_dir'], model_file))
    #Saving the loss files in an output directory to analyse for hyperparameter exploration
    torch.save({'losses_train': losses_train},
                   os.path.join(parameter_dict['output_dir'], metrics_file))

#Defining the dictionary for the performing hyperparameter exploration on the parameters 
training_parameters = {'embedding_length':512, 
                        #Selecting the embedding length 
                       'num_hiddens':512,
                        #Setting the number of hidden units in hidden layers 
                       'learning_rate':1e-3,
                        #Setting the initial learning rate 
                       'num_epochs':100,
                        #Running the model for num_epochs 
                       'num_layers':5}
                       #Defining the number of layers for the RNN architecture

#Defining the vocabulary
vocabulary = get_vocab(vocabulary_path)
#Defining the CNN architecture model
encoding_architecture = Resnet(training_parameters['embedding_length'])
#Defining the RNN architecture model 
decoding_architecture = RNN(training_parameters['embedding_length'],
              training_parameters['num_hiddens'],
              len(vocabulary),
              training_parameters['num_layers'])

#Defining the loss function as cross entropy 
loss_function = nn.CrossEntropyLoss()
#Collecting the encoding_architecture and decoding_architecture parameters together 
collected_params = list(decoding_architecture.parameters()) + list(encoding_architecture.linear.parameters()) + list(encoding_architecture.batchnorm.parameters())
#Defining the optimizer (ADAM/SGD with momentum)
optimizer = torch.optim.SGD(collected_params, lr = 0.01, momentum = 0.9)

#Transfering the models to the Blue Waters 
encoding_architecture.cuda()
decoding_architecture.cuda()

#Function to train the models
def train_model(train_data_loader, encoding_architecture, decoding_architecture, parameter_dict, resume_training):
    #If training from pretrained models, loading the CNN, RNN and optimizer states 
    if resume_training:
        state_dict = torch.load(os.path.join(parameter_dict['output_dir'], 'model_'+str(start_epoch+1)+'.ckpt'))
        #Loading the encoder
        encoding_architecture.load_state_dict(state_dict['encoder_state_dict'])
        #Loading the decoder
        decoding_architecture.load_state_dict(state_dict['decoder_state_dict'])
        #Loading the optimizer
        optimizer.load_state_dict(state_dict['optimizer'])
    #Setting the encoder model in the training mode
    encoding_architecture.train()
    #Setting the decoder model in the training mode
    decoding_architecture.train()
    #print("Models loaded")
    for epoch in range(100):
        #print(epoch)
        #Defining the list of the training losses over steps 
        train_loss_list = []
        #Enumerating through the Training Data Loader
        for index, (dataset_image, image_caption, caption_length) in enumerate(train_data_loader, start = 0):
            #print(step)
            #Converting the image and the corresponding caption to Pytorch Variables 
            #and sending to Blue Waters 
            dataset_image = Variable(dataset_image).cuda()
            image_caption = Variable(image_caption).cuda()
            #print("Data done")
            target_image_caption = nn.utils.rnn.pack_padded_sequence(image_caption, caption_length, batch_first=True)[0]
            #Initializing the optimizer
            optimizer.zero_grad()
            #Forward pass of the encoder model to retrieve the CNN features 
            CNN_feature = encoding_architecture(dataset_image)
            #print("Encoded")
            #Forward pass of the decoder model to retrieve the tokenized sentence 
            RNN_tokenized_sentence = decoding_architecture(CNN_feature, image_caption, caption_length)
            #print("Decoded")
            loss_value = loss_function(RNN_tokenized_sentence, target_image_caption)
            #Appending the training loss to the list 
            train_loss_list.append(loss_function(RNN_tokenized_sentence, target_image_caption).data[0])
            #Backward propagation of the loss function
            loss_value.backward()
            #print("Loss done")  
            #Taking a step for the optimizer and updating the parameters         
            optimizer.step()           
            #print("Checkpointing")  
            #Saving the checkpoint for the model every 5000 steps           
            if index%5000 == 0:
                checkpoint(encoding_architecture, decoding_architecture, optimizer, epoch, index, train_loss_list, parameter_dict)           
            if index%500 == 0:
                print('For Epoch: %d, %d the loss value is %0.2f '% (epoch+1, index, loss_value))
        print('For Epoch: %d, the loss value is %0.2f '% (epoch+1, np.mean(train_loss_list)))
        checkpoint(encoding_architecture, decoding_architecture, optimizer, epoch, index, train_loss_list, parameter_dict)

#Loading the train loader 
train_data_loader = get_data_loader(annotations_path=os.path.join(parameter_dict['data_dir'], 
                                        parameter_dict['train_ann_path']),
                                    data_path=os.path.join(parameter_dict['data_dir'], 
                                        parameter_dict['train_img_dir']),
                                    data_transforms=get_data_transforms(),
                                    parameter_dict=parameter_dict,
                                    vocabulary=vocabulary)
#Training the model 
train_model(decoding_architecture=decoding_architecture, encoding_architecture=encoding_architecture,parameter_dict=parameter_dict,
    resume_training=False,train_data_loader=train_data_loader)
