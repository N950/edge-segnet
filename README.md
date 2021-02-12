# EdgeSegNet - Pytorch

An implementation of [EdgeSegNet](https://arxiv.org/abs/1905.04222) in Pytorch 
[CamSeg01](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/) dataset is used for training

Repo containes :

* [EdgeSegNet.ipynb](https://github.com/N950/edge-segnet/blob/master/EdgeSegNet.py) Notebook taking care of downloading the dataset, training, plotting and prediction examples.

* [train.py](https://github.com/N950/edge-segnet/blob/master/train.py) entry point script which does all the downloading and setting up of the dataset, training and plotting graphs and prediction example

* [EdgeSegNet.py](https://github.com/N950/edge-segnet/blob/master/EdgeSegNet.py) is currently implemented in the exact architecture detailed in the paper, but could be modified easily. 
* [NetworkModules.py](https://github.com/N950/edge-segnet/blob/master/NetworkModules.py) custom modules are also implemented as Pytorch modules.
* [CamSeqDataset.py](https://github.com/N950/edge-segnet/blob/master/CamSeqDataset.py) a Pytorch Dataset for CamSeg01, downloads and unzips imgs.  
* [dataset_backup](https://github.com/N950/edge-segnet/tree/master/dataset_backup) 
# `train.py`
Default params will yield 90% val accuracy withing 40-50 epochs, 2 min on colab gpu 
```
usage: train.py [-h] [--learning-rate lr] [--batch-size B] [--n_epochs N]
                [--gamma G] [--scheduler-step S]

A Training script for EdgeSegNet :: https://arxiv.org/abs/1905.04222

optional arguments:
  -h, --help          show this help message and exit
  --learning-rate lr  Default 0.001, initial learning rate for the Adam optimizer, scheduled by StepLR
  --batch-size B      Default 16, Batch size for both train and validation, keep in mind the dataset has a total of 101 imgs only
  --n_epochs N        Default 50, number of training epochs
  --gamma G           Default 0.95, Multiplicative factor of learning rate decay
  --scheduler-step S  Default 25, Scheduler step, each S epochs learning rate is updated
```
