import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# sys.path.append(str(Path(__file__).resolve().parent))
# from local_diffusers.src.diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
# from local_diffusers.src.diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
# from local_diffusers.src.diffusers.schedulers.scheduling_pndm import PNDMScheduler

vae = AutoencoderKL.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="vae",
        use_safetensors=True
        )

tokenizer = CLIPTokenizer.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="tokenizer"
        )

text_encoder = CLIPTextModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="text_encoder",
        use_safetensors=True
        )

unet = UNet2DConditionModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="unet",
        use_safetensors=True
        )


scheduler = PNDMScheduler.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="scheduler"
        )

torch_device = "cpu" # "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

prompt = ["The godzilla is watching hello kitty doing her homework, they get along harmonious"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise
batch_size = len(prompt)

text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
        )
print("text_input:\n", text_input.input_ids)

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
print("text_embeddings:\n", text_embeddings)

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
print("uncond_embeddings:\n", uncond_embeddings)

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
print("conditional and unconditional embeddings", text_embeddings.shape, ":\n", text_embeddings)

torch.manual_seed(0)
latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=torch_device,
        )
latents = latents * scheduler.init_noise_sigma
print("random noise latents", latents.shape, ":\n", latents)

scheduler.set_timesteps(num_inference_steps)

print("scheduler timesteps:", scheduler.timesteps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
    print("latent_model_input", latent_model_input.shape, "\n", latent_model_input)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    print("timestep", t, "noise_pred", noise_pred.shape, ":\n", noise_pred)

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    print("timestep", t, "previous noisy latents", latents.shape, ":\n", latents)

# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)
image.save("./output-py.png")
