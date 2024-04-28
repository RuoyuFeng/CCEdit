import torch.nn as nn

from ...util import append_dims, instantiate_from_config


class Denoiser(nn.Module):
    def __init__(self, weighting_config, scaling_config):
        super().__init__()

        self.weighting = instantiate_from_config(weighting_config)
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def __call__(self, network, input, sigma, cond):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        # import pdb; pdb.set_trace()
        # import torchvision, einops
        # tmp = einops.rearrange(input, 'b c t h w -> (b t) c h w')
        # torchvision.utils.save_image(tmp[:,:3], 'tmp.png', normalize=True)
        '''
            input * c_in: noised_input multiplied by the coefficient of the corresponding t. (not sure)
            c_in: torch.Size([2, 1, 1, 1, 1]); 0.0683, 0.0683
            c_noise: the step t. e.g., tensor([451], device='cuda:0')
            cond: the condition. e.g., cond['crossattn']: [1, 77, 1024]
            c_out: e.g., -1.3762. Don't know why multiply this and why it's negative.
            c_skip: e.g., 1.0. Don't know why multiply this.
        '''
        return network(input * c_in, c_noise, cond) * c_out + input * c_skip


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        weighting_config,
        scaling_config,
        num_idx,
        discretization_config,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
    ):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise