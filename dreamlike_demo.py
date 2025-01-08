import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from tqdm.auto import tqdm

vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        use_safetensors=True
        )

tokenizer = CLIPTokenizer.from_pretrained(
        "dreamlike-art/dreamlike-photoreal-2.0",
        subfolder="tokenizer"
        )

text_encoder = CLIPTextModel.from_pretrained(
        "dreamlike-art/dreamlike-photoreal-2.0",
        subfolder="text_encoder",
        use_safetensors=True
        )

unet = UNet2DConditionModel.from_pretrained(
        "dreamlike-art/dreamlike-photoreal-2.0",
        subfolder="unet",
        use_safetensors=True
        )

from diffusers import UniPCMultistepScheduler

scheduler = UniPCMultistepScheduler.from_pretrained(
        "dreamlike-art/dreamlike-photoreal-2.0",
        subfolder="scheduler"
        )

torch_device = "cpu" # "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

# from diffusers import AutoPipelineForText2Image
#
# pipe_txt2img = AutoPipelineForText2Image.from_pretrained(
#     "dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16, use_safetensors=True
# ).to("cuda")
#
# generator = torch.Generator(device="cuda").manual_seed(37)
# image = pipe_txt2img(prompt, generator=generator).images[0]
# image

prompt = ["godzilla is watching kitty doing homework"]
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
print(text_input.input_ids)

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

print(text_embeddings)

latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=torch_device,
        )
latents = latents * scheduler.init_noise_sigma


scheduler.set_timesteps(num_inference_steps)


for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    breakpoint()
    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample
