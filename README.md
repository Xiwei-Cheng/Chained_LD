# On the Mode-Seeking Properties of Langevin Dynamics

This repo contains the official implementation for the paper On the Mode-Seeking Properties of Langevin Dynamics by Xiwei Cheng, Kexin Fu, and Farzan Farnia. 


## Getting started

To train a neural network for chained Langevin dynamics on the original and flipped images from the MNIST dataset, run
```
python3 main.py  --runner ChainedRunner  --doc chained_mnist_flip  --config chained_mnist_flip.yml
```
Then the model will be trained according to the configuration files in `configs/chained_mnist_flip.yml`, and the log files will be stored in `run/logs/chained_mnist_flip`. 

To generate images using chained Langevin dynamics with 30000 iterations, run
```
python3 main.py  --test  --test_iter 30000  --runner ChainedRunner  --doc chained_mnist_flip  --config chained_mnist_flip.yml
```
Then the generated samples will be saved in `run/logs/chained_mnist_flip/images_iter30000`. 


## References

The implementation is based on the paper [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) ([code](https://github.com/ermongroup/ncsn)).
