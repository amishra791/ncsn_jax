import jax.random as jr
from torchvision.transforms.functional import to_pil_image
import numpy as np
import jax.numpy as jnp


def save_and_log(sample, name=None):
    sample = jnp.clip(sample, 0, 1)
    sample = np.array(sample)
    pil_img = to_pil_image(sample[0])
    pil_img.save(f"{name}.png")


def langevin_sampling(model, shape, sigmas, rngs, x0_bhwd=None):
    model.eval()
    eps = 2e-5
    T = 100

    sigma_max = sigmas[0]
    sigma_min = sigmas[-1]
    assert sigma_max == jnp.max(sigmas)
    assert sigma_min == jnp.min(sigmas)

    if x0_bhwd is not None:
        z0 = jr.normal(rngs.sampling(), shape=x0_bhwd.shape, dtype=x0_bhwd.dtype)
        curx_bhwd = x0_bhwd + sigma_max * z0
    else:
        curx_bhwd = jr.normal(rngs.sampling(), shape=shape) * sigma_max
    for _, sigma in enumerate(sigmas): 
        sigma_viewed = jnp.full((shape[0], 1, 1, 1), sigma)
        cur_alpha = eps * (sigma / sigma_min) ** 2
        
        for _ in range(T):
            z_bhwd = jr.normal(rngs.sampling(), shape=curx_bhwd.shape)
            cur_scores = model(curx_bhwd, sigma_viewed)
            curx_bhwd = curx_bhwd + 0.5 * cur_alpha * cur_scores + jnp.sqrt(cur_alpha) * z_bhwd
    
    return curx_bhwd
