from model import NCSNV2
from data import get_mnist_dataloader
from flax import nnx
import jax.random as jr
import optax
import jax
import orbax.checkpoint as ocp
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
import jax.numpy as jnp
from langevin_sampling import langevin_sampling
import os
import random
import os

samples_directory = "mnist_samples"
try:
    # Create the directory in the current working directory
    os.makedirs(samples_directory, exist_ok=True)
    print(f"Directory '{samples_directory}' ensured to exist.")
except OSError as e:
    print(f"Error creating directory: {e}")

rngs = nnx.Rngs(0, params=1, sampling=2)

ckpt_dir = '/tmp/ncsn_chekcpoint/'
ckpt_state = ckpt_dir + "state"
checkpointer = ocp.StandardCheckpointer()

# Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
abstract_model = nnx.eval_shape(lambda: NCSNV2(in_features=1, rngs=nnx.Rngs(0)))
graphdef, abstract_state = nnx.split(abstract_model)

state_restored = checkpointer.restore(ckpt_state, abstract_state)
print('NNX State restored')

# The model is now good to use!
model = nnx.merge(graphdef, state_restored)

sigmas = jnp.geomspace(1.0, 0.01, num=50)

num_samples = 64
sampled_img_tensor = langevin_sampling(model, (num_samples, 28, 28, 1), sigmas, rngs, None)
sampled_img_tensor = jnp.clip(sampled_img_tensor, 0, 1)
sampled_img_tensor = np.array(sampled_img_tensor)

for i in range(num_samples):
    pil_img = to_pil_image(sampled_img_tensor[i])
    pil_img.save(os.path.join(samples_directory, f"sampled_image_{i}.png"))
