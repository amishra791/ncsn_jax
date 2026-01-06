from model import NCSNV2
from data import get_mnist_dataloader
from flax import nnx
import jax.random as jr
import optax
import orbax.checkpoint as ocp
import jax.numpy as jnp


@nnx.jit
def train_step(model, optimizer, x_bhwd, sigmas, sigma_idxs, z_bhwd):
    sigmas_b = sigmas[sigma_idxs]
    sigmas_bhwd = jnp.reshape(sigmas_b, shape=(-1, 1, 1, 1))
  
    noise = z_bhwd * sigmas_bhwd
    perturbed_x_bhwd = x_bhwd + noise

    target = -1 / (sigmas_bhwd ** 2) * noise
    target = jnp.reshape(target, shape=(target.shape[0], -1))

    def loss_fn(model):
        scores = model(perturbed_x_bhwd, sigmas_bhwd)
        scores = jnp.reshape(scores, shape=(scores.shape[0], -1))

        normed_diff = jnp.mean((scores - target) ** 2, axis=-1)
        loss_b = 0.5 * sigmas_b ** 2 * normed_diff

        return jnp.mean(loss_b)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)


    return loss

# setup checkpoint dir
ckpt_dir = ocp.test_utils.erase_and_create_empty('/tmp/ncsn_chekcpoint')


# get nnx random keys
# params are for initializing the model
# default is for drawing noise vector for training
rngs = nnx.Rngs(0, params=1)



batch_size = 128
sigmas = jnp.geomspace(1.0, 0.01, num=50)

dataloader = get_mnist_dataloader(batch_size=batch_size)

# set up model and optimizer
model = NCSNV2(in_features=1, rngs=rngs)

steps_per_epoch = 500
num_epochs = 30
total_steps = steps_per_epoch * num_epochs

warmup_epochs = 2
warmup_steps = warmup_epochs * steps_per_epoch

peak_lr = 3e-5
end_lr  = 3e-6

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=peak_lr,
    warmup_steps=warmup_steps,
    decay_steps=total_steps - warmup_steps,
    end_value=end_lr,
)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(lr_schedule),
)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
model.train()

for epoch_idx in range(num_epochs):
    for i, (x_bhwd, _) in enumerate(dataloader):
        sigma_idxs = jr.randint(rngs(), shape=(x_bhwd.shape[0],), minval=0, maxval=len(sigmas))
        z_bhwd = jr.normal(rngs(), shape=x_bhwd.shape)
        loss = train_step(model, optimizer, x_bhwd, sigmas, sigma_idxs, z_bhwd)
        print(f"Epoch: {epoch_idx}, i: {i}, Loss: {loss}")

_, state = nnx.split(model)
# nnx.display(state)
ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
ckptr.save(ckpt_dir / 'state', args=ocp.args.StandardSave(state))
ckptr.wait_until_finished()
print("all done")
