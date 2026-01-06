# ncsn_jax

This is an unofficial implementation of score-matching models in JAX, as outlined in these two papers: 
* [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600)
* [Improved Techniques for Training Score-Based Generative Models](https://arxiv.org/pdf/2006.09011)


There are some important implementation details that might differ from these two papers: 
* We divide the L2 loss function by the dimension of the image space due to exploding gradients
* We use a LR scheduler
* For MNIST, we used a 4-cascaded RefineNet. More details [here](https://arxiv.org/abs/1611.06612)

`layers.py` and `model.py` contain the implementation of the underlying U-Net diffusion model. 

`main.py` contains the training code

`langevin_sampling.py` contains the code for Langevin sampling

`load_and_sample.py` has a script to load a saved model and generate sampled images
