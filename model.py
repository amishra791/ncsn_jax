from flax import nnx
import jax
import jax.numpy as jnp
from flax.linen.pooling import avg_pool as avg_pool
from layers import ResNetBlock, RefineNetBlock, pytorch_conv_bias_init, kernel_init


class NCSNV2(nnx.Module):

    # predefine a 4 cascaded refine net
    def __init__(self, in_features, rngs: nnx.Rngs):
        begin_bias_init = pytorch_conv_bias_init(in_channels=in_features, kernel_size=(3, 3), groups=1)
        self.begin_conv = nnx.Conv(
            in_features, 
            out_features=128, 
            kernel_size=(3,3), 
            padding='SAME', 
            strides=(1,1), 
            kernel_init=kernel_init,
            bias_init=begin_bias_init,
            rngs=rngs
        )

        self.res_1 = ResNetBlock(
            in_features=128,
            out_features=128, 
            strides=(1,1),
            rngs=rngs
        )
        self.res_2 = ResNetBlock(
            in_features=128,
            out_features=256, 
            strides=(2,2),
            rngs=rngs
        )
        self.res_3 = ResNetBlock(
            in_features=256,
            out_features=512, 
            strides=(2,2),
            rngs=rngs
        )
        self.res_4 = ResNetBlock(
            in_features=512,
            out_features=1024,
            strides=(2,2),
            rngs=rngs
        )

        self.refine_4 = RefineNetBlock(channels_list=[1024], rngs=rngs)
        self.refine_3 = RefineNetBlock(channels_list=[512, 1024], rngs=rngs)
        self.refine_2 = RefineNetBlock(channels_list=[256, 512], rngs=rngs)
        self.refine_1 = RefineNetBlock(channels_list=[128, 256], rngs=rngs)

        final_bias_init = pytorch_conv_bias_init(in_channels=128, kernel_size=(3, 3), groups=1)
        self.final_conv = nnx.Conv(
            128, 
            out_features=in_features, 
            kernel_size=(1,1), 
            padding='SAME', 
            strides=(1,1), 
            kernel_init=kernel_init,
            bias_init=final_bias_init,
            rngs=rngs
        )


    def __call__(self, x_bhcw, sigma_b):
        out = self.begin_conv(x_bhcw)

        res_1_out = self.res_1(out)
        res_2_out = self.res_2(res_1_out)
        res_3_out = self.res_3(res_2_out)
        res_4_out = self.res_4(res_3_out)

        refine_4_out = self.refine_4([res_4_out])
        refine_3_out = self.refine_3([res_3_out, refine_4_out])
        refine_2_out = self.refine_2([res_2_out, refine_3_out])
        refine_1_out = self.refine_1([res_1_out, refine_2_out])

        final_conv_out = self.final_conv(refine_1_out)

        return final_conv_out / sigma_b