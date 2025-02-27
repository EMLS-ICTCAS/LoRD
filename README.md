# Low-redundancy Distillation for Continual Learning

## Introduction
This the training and evaluation code for our work "Low-redundancy Distillation for Continual Learning".


## Setup

+ Use `python main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters for each of the evaluation setting from the paper.
+ To reproduce the results in the paper run the following  

    `python main.py --dataset <dataset> --model <model> --buffer_size <buffer_size> --load_best_args`

 Examples:

    python main.py --dataset seq-cifar10 --model lord --buffer_size 500 --load_best_args
    
    python main.py --dataset seq-tinyimg --model lord --buffer_size 500 --load_best_args
   
    python main.py --dataset seq-cifar100 --model lord --buffer_size 500 --load_best_args
    
 

## Acknowledgements :
We extended the original repo [DER++](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) with our method.
We extend our gratitude to the authors for their support and for providing the research community with the Mammoth framework.
