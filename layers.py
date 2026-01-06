from flax import nnx
import jax
import jax.numpy as jnp
from flax.linen.pooling import max_pool as max_pool
from flax.linen.pooling import avg_pool as avg_pool

def pytorch_conv_bias_init(in_channels, kernel_size, groups=1):
    kh, kw = kernel_size
    fan_in = (in_channels // groups) * kh * kw
    bound = 1.0 / jnp.sqrt(fan_in)

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init

kernel_init = jax.nn.initializers.variance_scaling(
    scale=1/3,
    mode="fan_in",
    distribution="uniform",
)

class InstanceNormalization(nnx.Module):
    """
    Instance Normalization (no running statistics).

    Normalizes each sample independently, per-channel, over spatial axes only.
    Default assumes NHWC inputs: (B, H, W, C).

    y = (x - mean) / sqrt(var + eps) * gamma + beta
    where mean/var are computed over (H, W) for each (B, C).

    Notes:
      - No running mean/var are stored or updated (unlike BatchNorm).
      - Fully JIT-compatible; shape must be stable across traces if you JIT.
    """

    def __init__(self, num_channels, rngs: nnx.Rngs):
        self.eps = 1e-5
        self.num_channels = num_channels

        self.gamma = nnx.Param(jnp.ones((self.num_channels,), dtype=jnp.float32))
        self.beta = nnx.Param(jnp.zeros((self.num_channels,), dtype=jnp.float32))

    def __call__(self, x_bhwc: jnp.ndarray) -> jnp.ndarray:

        assert x_bhwc.ndim == 4, f"expected (B,H,W,C), got {x_bhwc.shape}"
        assert x_bhwc.shape[-1] == self.num_channels, (
            f"channel mismatch: x has C={x_bhwc.shape[-1]}, module has {self.num_channels}"
        )

        # Compute per-instance, per-channel mean/var over spatial axes.
        mean = jnp.mean(x_bhwc, axis=(1, 2), keepdims=True)
        var = jnp.mean((x_bhwc - mean) ** 2, axis=(1, 2), keepdims=True)
        inv_std = jax.lax.rsqrt(var + self.eps)  # stable + faster than 1/sqrt

        y = (x_bhwc - mean) * inv_std

        gamma = self.gamma.value.reshape((1, 1, 1, self.num_channels))
        beta = self.beta.value.reshape((1, 1, 1, self.num_channels))
        y = y * gamma + beta

        return y

    
class ResNetBlock(nnx.Module):
    """
    Pre-norm ResNet block (NHWC):

      y = Conv3x3(stride)(ELU(IN(x)))
      y = Conv3x3(1)(ELU(IN(y)))
      out = skip(x) + y

    where:
      skip = Identity if shapes match
           = Conv1x1(stride) otherwise
    """

    def __init__(
        self,
        in_features,
        out_features,
        strides,
        rngs: nnx.Rngs,
    ):
        self.norm1 = InstanceNormalization(in_features, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=strides,
            padding="SAME",
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.norm2 = InstanceNormalization(out_features, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        needs_proj = (strides != (1, 1)) or (in_features != out_features)

        self.proj = (
            nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(1, 1),
                strides=strides,
                padding="SAME",
                use_bias=False,
                kernel_init=kernel_init,
                rngs=rngs,
            )
            if needs_proj
            else nnx.identity
        )

    def __call__(self, x_bhwc: jnp.ndarray) -> jnp.ndarray:
        y = self.norm1(x_bhwc)
        y = nnx.elu(y)
        y = self.conv1(y)

        y = self.norm2(y)
        y = nnx.elu(y)
        y = self.conv2(y)

        return self.proj(x_bhwc) + y

class RCU(nnx.Module):
    """
    Residual Convolution Unit (RefineNet-style), with ELU activations and no normalization.

    Structure:
      y = Conv3x3(ELU(x))
      y = Conv3x3(ELU(y))
      out = x + y
    """

    def __init__(self, features: int, rngs: nnx.Rngs):
        bias_init = pytorch_conv_bias_init(in_channels=features, kernel_size=(3, 3), groups=1)
        self.conv1 = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )

    def __call__(self, x_bhwc: jnp.ndarray) -> jnp.ndarray:
        y = nnx.elu(x_bhwc)
        y = self.conv1(y)

        y = nnx.elu(y)
        y = self.conv2(y)

        return x_bhwc + y
    
class TwoRCU(nnx.Module):
    """A simple 2xRCU stack operating at a fixed channel dimension."""
    def __init__(self, features, rngs: nnx.Rngs):
        self.rcu1 = RCU(features, rngs=rngs)
        self.rcu2 = RCU(features, rngs=rngs)

    def __call__(self, x_bhwc: jnp.ndarray) -> jnp.ndarray:
        return self.rcu2(self.rcu1(x_bhwc))
    
class AdaptiveConvUnit(nnx.Module):
    """
    Applies a dedicated 2xRCU stack to each input tensor.

    Inputs:
      - xs: sequence/tuple of tensors, each NHWC
    Construction:
      - channel_list[i] = channels of xs[i]
      - builds per-input TwoRCU(channel_list[i])

    Returns:
      - tuple of tensors, same length/order as inputs
    """

    def __init__(self, channel_list, rngs: nnx.Rngs):
        self.channel_list = list(channel_list)
        self.units = nnx.List([TwoRCU(ch, rngs=rngs) for ch in self.channel_list])

    def __call__(self, xs):
        assert len(xs) == len(self.units), (
            f"Expected {len(self.units)} inputs, got {len(xs)}"
        )

        ys = []
        for i, (x, unit, ch) in enumerate(zip(xs, self.units, self.channel_list)):
            assert x.shape[-1] == ch, (
                f"Input {i} channel mismatch: expected C={ch}, got C={x.shape[-1]}"
            )
            ys.append(unit(x))

        return tuple(ys)

class MultiResolutionFusion(nnx.Module):
    """
    Multi-Resolution Fusion (MRF):

    - Projects each input to min(channel_list) via a 3x3 conv (when >1 input).
    - Upsamples all inputs to the maximum H and W among the inputs using bilinear resize.
    - Sums the aligned tensors.

    Special case:
      - If only one input is expected, uses Identity (clean/no-op).
    """

    def __init__(self, channel_list, rngs: nnx.Rngs):
        self.channel_list = list(channel_list)

        if len(self.channel_list) == 1:
            self.min_channels = self.channel_list[0]
            self.proj = nnx.identity
            self.proj_convs = None
            return

        self.min_channels = min(self.channel_list)

        # Per-input projection to min_channels
        self.proj_convs = nnx.List([
            nnx.Conv(
                in_features=ch,
                out_features=self.min_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,   # no norm in RefineNet blocks
                kernel_init=kernel_init,
                bias_init=pytorch_conv_bias_init(in_channels=ch, kernel_size=(3, 3), groups=1),
                rngs=rngs,
            )
            for ch in self.channel_list
        ])

    def __call__(self, xs) -> jnp.ndarray:
        assert len(xs) == len(self.channel_list), (
            f"Expected {len(self.channel_list)} inputs, got {len(xs)}"
        )

        # Single-input fast path: no projection, no resize, no sum.
        if len(self.channel_list) == 1:
            return self.proj(xs[0])

        # Determine target spatial resolution (max H, max W across inputs).
        target_h = max(x.shape[1] for x in xs)
        target_w = max(x.shape[2] for x in xs)

        ys = []
        for i, (x, ch, conv) in enumerate(zip(xs, self.channel_list, self.proj_convs)):
            assert x.ndim == 4, f"Input {i} must be BHWC, got {x.shape}"
            assert x.shape[-1] == ch, (
                f"Input {i} channel mismatch: expected C={ch}, got C={x.shape[-1]}"
            )

            y = conv(x)  # (B, H, W, min_channels)

            if (y.shape[1] != target_h) or (y.shape[2] != target_w):
                # Bilinear interpolation in BHWC
                y = jax.image.resize(
                    y,
                    shape=(y.shape[0], target_h, target_w, y.shape[-1]),
                    method="linear",
                )

            ys.append(y)

        return jnp.sum(jnp.stack(ys, axis=0), axis=0)

class CRPBlock(nnx.Module):
    """
    Chained Residual Pooling sub-block:
      5x5 avg_pool (stride 1) -> 3x3 conv

    NHWC input: (B, H, W, C)
    """

    def __init__(self, num_channels, rngs: nnx.Rngs):
        self.num_channels = num_channels
        self.conv = nnx.Conv(
            in_features=num_channels,
            out_features=num_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,  # no norm in RefineNet blocks
            kernel_init=kernel_init,
            bias_init=pytorch_conv_bias_init(in_channels=num_channels, kernel_size=(3, 3), groups=1),
            rngs=rngs,
        )

    def __call__(self, x_bhwc):
        assert x_bhwc.shape[-1] == self.num_channels, (
            f"CRPBlock expects C={self.num_channels}, got C={x_bhwc.shape[-1]}"
        )

        y = max_pool(
            x_bhwc,
            window_shape=(5, 5),
            strides=(1, 1),
            padding="SAME",
        )
        y = self.conv(y)
        return y

class CRP(nnx.Module):
    """
    Chained Residual Pooling (2-stage).

    Typical RefineNet CRP accumulates residual sums:
      x0 = x
      x1 = x0 + CRPBlock(x0)
      x2 = x1 + CRPBlock(x1)
      return x2

    (No normalization; activation is handled outside in the RefineNet block.)
    """

    def __init__(self, num_channels: int, rngs: nnx.Rngs):
        self.num_channels = num_channels
        self.block1 = CRPBlock(num_channels, rngs=rngs)
        self.block2 = CRPBlock(num_channels, rngs=rngs)

    def __call__(self, x_bhwc):
        assert x_bhwc.shape[-1] == self.num_channels, (
            f"CRP expects C={self.num_channels}, got C={x_bhwc.shape[-1]}"
        )

        y1 = self.block1(x_bhwc)
        y2 = self.block2(y1)
        return x_bhwc + y1 + y2

class RefineNetBlock(nnx.Module):
    """
    RefineNet block (component-level), assembled from:
      1) AdaptiveConvUnit (per-input 2xRCU)
      2) MultiResolutionFusion (project-to-min-channels + upsample-to-max-HW + sum)
      3) CRP (2-stage chained residual pooling) operating at min(channels_list)
      4) Output RCU (single RCU) at min(channels_list)

    Inputs:
      - xs: sequence/tuple of NHWC tensors whose channel dims match channels_list
    Output:
      - single NHWC tensor at spatial size = max(H,W) among inputs, channels = min(channels_list)
    """

    def __init__(self, channels_list, rngs: nnx.Rngs):
        self.channels_list = list(channels_list)
        self.out_channels = min(self.channels_list)

        self.acu = AdaptiveConvUnit(self.channels_list, rngs=rngs)
        self.mrf = MultiResolutionFusion(self.channels_list, rngs=rngs)
        self.crp = CRP(self.out_channels, rngs=rngs)
        self.out_rcu = RCU(self.out_channels, rngs=rngs)

    def __call__(self, xs):
        # 1) per-input adaptation
        xs = self.acu(xs)

        # 2) multi-resolution fusion to min-channels at max spatial resolution
        y = self.mrf(xs)

        # 3) chained residual pooling
        y = self.crp(y)

        # 4) output RCU
        y = self.out_rcu(y)

        return y
