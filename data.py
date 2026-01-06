import numpy as np
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST
import jax.numpy as jnp
from torchvision import transforms

def numpy_collate(batch):
    """
    Collate function specifies how to combine a list of data samples into a batch.
    default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
    """
    return tree_map(np.asarray, default_collate(batch))

def flatten_and_cast(pic):
    """Convert PIL image to flat (1-dimensional) numpy array."""
    return np.array(pic.permute(1, 2, 0), dtype=jnp.float32)

def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def get_mnist_dataset_and_labels(batch_size=64):
    mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    flatten_and_cast
    ]))

    train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
    train_labels = one_hot(np.array(mnist_dataset.train_labels), 10)

    return train_images, train_labels

def get_mnist_dataloader(batch_size=64, shuffle=False):
    transform_composer = transforms.Compose([transforms.ToTensor(), flatten_and_cast])
    mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=transform_composer)
    # Create pytorch data loader with custom collate function
    training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate, shuffle=shuffle)

    return training_generator