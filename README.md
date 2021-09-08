# Project Setup

> install Conda

## Create Environment

    conda env create -f environment.yml

## Terms
* input: images directory
* input_lb: label csv file path
* out: output directory to save the trained model results
* pretrained: a .pth pretrained model,
* op: operation type
* seed: determine seed
* num_worker: number of thread for prefetching the next batch. (improve the loading speed)
* shuffle: shuffle the data
* train_size: training size
* test_size: testing size
* weight_decay: weight decay parameter
* lr: learning rate
* width: image width
* height: image height
* channel: image channel
* beta1: Adam optimizer parameter
* gen_feature_map: generator hidden feature map
* disc_feature_map: discriminator hidden feature map
* epochs: number of epochs
* batch: batch size
* clip: parameter clip range
* c_repeat: number of discriminator updating
* c_lambda: gradient penalty weight
* sigma: noise scale coefficient (differential privacy)
* latent_dim: latent dimention
* device: use cpu or gpu, 1 ~ gpu and 0 ~ cpu
* alpha: mix up coefficient
* mix_up: enable mix up utility on training gan
* dp: enable dp utility on training gan
* tensorboard: enable tensorboard for seeing result
* posix: title for saving classifier loss 

## Commands
### Train Generator
Train GAN network without dp and mix up

    python manager.py --input /path/dataset --op train_w_gan --out /path/to/save --num_worker 4 --epochs 10

Train GAN network with dp and without mix up

    python manager.py --input /path/dataset --op train_w_gan --out /path/to/save --num_worker 4 --epochs 10 --dp

Train GAN network without dp and with mix up

    python manager.py --input /path/dataset --op train_w_gan --out /path/to/save --num_worker 4 --epochs 10 --mix_up

Train GAN network with dp and with mix up

    python manager.py --input /path/dataset --op train_w_gan --out /path/to/save --num_worker 4 --epochs 10 --dp --mix_up


### Train Classifier

Train classifier with no pretrained model

    python manager.py --input /path/dataset --op train_classifier --out /path/to/save --num_worker 4 --epochs 20

Train classifier with pretrained model

     python manager.py --input /path/dataset --op train_classifier --out /path/to/save --num_worker 4 --epochs 20 --pretrained /path/to/model.pth

### Formula

    for epoch = 1,2, ... epochs
        for batch1,batch2 in dataloader
            