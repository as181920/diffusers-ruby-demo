import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

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

torch_device = "cuda"
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
