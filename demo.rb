require "chunky_png"
require "torch-rb"
require "onnxruntime"
require "tokenizers"
require "debug"
require_relative "./ddim_scheduler"
require_relative "./pndm_scheduler"

# Set onnxruntime device
if Torch::CUDA.available? && ENV.fetch("DEVICE", "cpu").eql?("cuda")
  OnnxRuntime.ffi_lib = ENV.fetch("ONNX_lib", OnnxRuntime.ffi_lib)
  DEVICE = "cuda".freeze
  PROVIDERS = %w[CUDAExecutionProvider].freeze
else
  DEVICE = "cpu".freeze
  PROVIDERS = %w[CPUExecutionProvider].freeze
end

# model:
# https://huggingface.co/docs/diffusers/tutorials/autopipeline
#
# run steps:
# https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#deconstruct-the-stable-diffusion-pipeline

# Create text tokens
prompt = ["godzilla is watching kitty doing homework"]
batch_size = prompt.length

# Create text tokens
tokenizer = Tokenizers.from_pretrained("openai/clip-vit-large-patch14") # openai/clip-vit-base-patch32
tokenizer.enable_padding(length: 77, pad_id: 49407)
tokenizer.enable_truncation(77)
text_tokens = tokenizer.encode_batch(prompt)
text_ids = Torch.tensor(text_tokens.map(&:ids))
print "text_tokens(#{text_ids.shape}):\n", text_ids, "\n"

# Create text embeddings
text_encoder = OnnxRuntime::Model.new("./onnx/text_encoder/model.onnx", providers: PROVIDERS) # name: openai/clip-vit-large-patch14
text_embeddings = Torch.no_grad do
  text_encoder
    .predict({ input_ids: text_ids }) # Shape: 1x77
    .then { |h| Torch.tensor(h["last_hidden_state"]) } # Shape: 1x77x768
end
print "text_embeddings(#{text_embeddings.shape}):\n", text_embeddings, "\n"

# Generate the unconditional text embeddings which are the embeddings for the padding token.
# max_length = text_ids.shape[-1] # 77
uncond_tokens = tokenizer.encode_batch([""] * batch_size)
uncond_ids = Torch.tensor(uncond_tokens.map(&:ids))
uncond_embeddings = text_encoder
  .predict({ input_ids: uncond_ids })
  .then { |h| Torch.tensor(h["last_hidden_state"]) } # Shape: 1x77x768

text_embeddings = Torch.cat([uncond_embeddings, text_embeddings])

# Create random noise
height = 512
width = 512
unet = OnnxRuntime::Model.new("./onnx/unet/model.onnx", providers: PROVIDERS)
channels_num = unet.inputs.detect{ |e| e[:name] == "sample" }[:shape][1]
generator = Torch::Generator.new.manual_seed(0) # Seed generator to create the initial latent noise
Torch.manual_seed(42)
latents = Torch.randn([batch_size, channels_num, height / 8, width / 8], generator:, device: DEVICE) # Shape: 1x4x64x64
# Set scheduler
# scheduler = DDIMScheduler.new(steps_offset: 1, timestep_spacing: "leading")
scheduler = PNDMScheduler.new(steps_offset: 1, timestep_spacing: "leading")
latents = latents * scheduler.init_noise_sigma # Scaling the input with the initial noise distribution, sigma
num_inference_steps = 25 # denoising steps
scheduler.num_inference_steps = num_inference_steps

# Denoise the image
guidance_scale = 7.5 # classifier-free guidance, determines how much weight should be given to the prompt when generating an image.
scheduler.timesteps.each do |timestep|
  print "scheduler timestep #{timestep} started at #{Time.now.strftime('%F %T')}\n"
  # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
  latent_model_input = Torch.cat([latents] * 2)
    .then { |input| scheduler.scale_model_input(input, timestep:) }

  # predict the noise residual
  noise_pred = Torch.no_grad do
    unet
      .predict({ sample: latent_model_input, timestep: Torch.tensor(timestep), encoder_hidden_states: text_embeddings })
      .then { |h| Torch.tensor(h["out_sample"]) } # Shape: 2x4x64x64
  end

  #########################################
  # Perform guidance
  noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
  noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

  # Compute the previous noisy sample x_t -> x_t-1
  latents = scheduler.step(noise_pred, timestep, latents)[:prev_sample]
end

# Decode the image

# scale and decode the image latents with vae
# vae_encoder = OnnxRuntime::Model.new("./onnx/vae_encoder/model.onnx", providers: PROVIDERS)
vae_decoder = OnnxRuntime::Model.new("./onnx/vae_decoder/model.onnx", providers: PROVIDERS)
latents = latents / 0.18215
image = Torch.no_grad do
  vae_decoder
    .predict({latent_sample: latents})
    .then { |h| Torch.tensor(h["sample"]) } # Shape: 1x3x512x512
end

# save image

# Step 1: (image / 2 + 0.5).clamp(0, 1)
image = ((image / 2.0) + 0.5).clip(0, 1)
# Step 2: 去除批次维度（假设 image 是 4D 张量）
# image = image[0, true, true, true] if image.ndim == 4
image = image[0] if image.ndim == 4
# Step 3: 调整维度顺序，从 (C, H, W) 到 (H, W, C)
image = image.permute(1, 2, 0)
# Step 4: 转换到 uint8 并放大到 [0, 255]
image = (image * 255).round.to(Torch.uint8)
# Step 5: 将数组转换为 PNG 图像并保存
output_height, output_width, _channels = image.shape
png = ChunkyPNG::Image.new(output_width, output_height)
height.times do |y|
  width.times do |x|
    r, g, b = image[y, x, 0..2].map(&:to_i) # 取 RGB 值
    png[x, y] = ChunkyPNG::Color.rgb(r, g, b)
  end
end

# 保存图像
png.save("./output.png")
