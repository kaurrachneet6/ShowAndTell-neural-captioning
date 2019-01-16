## Show and Tell: Neural Image Caption Generator
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