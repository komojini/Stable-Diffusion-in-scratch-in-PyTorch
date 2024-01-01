import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

# https://www.youtube.com/watch?v=ZBKpAp_6TGI

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8


def rescale(x, in_range, out_range, clamp=False):
    in_min, in_max = in_range
    out_min, out_max = out_range
    x -= in_min
    x *= (out_max - out_min) / (in_max - in_min)
    x += out_min
    if clamp:
        x = x.clamp(out_min, out_max)
    return x


def get_time_embedding(timestep: int) -> torch.Tensor:
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (160) -> (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 160) -> (1, 320)
    x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    return x


def generate(
        prompt: str, 
        negative_prompt: str, 
        input_image=None, 
        strength=0.8, 
        do_cfg=True, 
        cfg_scale=7.5, 
        sampler_name="ddpn", 
        n_inference_steps=50, 
        models={}, 
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
    ):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        
        generator = torch.Generator(device=device)

        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
            cond_context = clip(cond_tokens)

            # Convert the negative prompt into tokens using the tokenizer
            neg_tokens = tokenizer.batch_encode_plus([negative_prompt], padding="max_length", max_length=77).input_ids
            # (batch_size, seq_len)
            neg_tokens = torch.tensor(neg_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
            neg_context = clip(neg_tokens)

            # (2, seq_len, embed_dim)
            context = torch.cat([cond_context, neg_context])

        else:
            # Convert it into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, seq_len) -> (1, seq_len, embed_dim)
            context = clip(tokens)
        to_idle(context)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (height, width, channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # (height, width, channel) -> (batch_size, height, width, channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch_size, height, width, channel) -> (batch_size, channel, height, width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
        
            to_idle(encoder)
        
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)
        
        diffusion = models["diffusion"]
        diffusion.to(device)

        # 999 ... 0
        # 1000 ... 1
        # 1000 980 960, ... 0
        timesteps = tqdm(sampler.timesteps):
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise by the UNET 
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_neg = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_neg) + output_neg

            # Remove the noise predicted by the UNET from the latents            
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images
    



