require "torch"
require "onnxruntime"
require "tokenizers"
require "debug"

providers = if Torch::CUDA.available?
              %w[CUDAExecutionProvider]
            else
              %w[CPUExecutionProvider]
            end
device = "cpu" # use "cuda" with gpu, currently use cpu temporarily to handle gpu runtime error

# model:
# https://huggingface.co/docs/diffusers/tutorials/autopipeline
#
# run steps:
# https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#deconstruct-the-stable-diffusion-pipeline
#
# scheduler: there is no Ruby implementation yet.

# Create text tokens
prompt = ["godzilla is watching kitty doing homework"]
batch_size = prompt.length

# Create text tokens
tokenizer = Tokenizers.from_pretrained("openai/clip-vit-base-patch32")
tokenizer.enable_padding(length: 77, pad_id: 49407)
tokenizer.enable_truncation(77)
text_tokens = tokenizer.encode_batch(prompt)
text_ids = Torch.tensor(text_tokens.map(&:ids))
pp text_ids

# Create text embeddings
text_encoder = OnnxRuntime::Model.new("../onnx/text_encoder/model.onnx", providers:) # name: openai/clip-vit-large-patch14
text_embeddings = Torch.no_grad do
  text_encoder
    .predict({ input_ids: text_ids }) # Shape: 1x77
    .then { |h| Torch.tensor(h["last_hidden_state"]) } # Shape: 1x77x768
end
pp text_embeddings

# Generate the unconditional text embeddings which are the embeddings for the padding token.
# max_length = text_ids.shape[-1] # 77
uncond_tokens = tokenizer.encode_batch([""] * batch_size)
uncond_ids = Torch.tensor(uncond_tokens.map(&:ids))
uncond_embeddings = text_encoder.predict({ input_ids: uncond_ids })
  .then { |h| Torch.tensor(h["last_hidden_state"]) } # Shape: 1x77x768

text_embeddings = Torch.cat([uncond_embeddings, text_embeddings])

# Create random noise
height = 512
width = 512
unet = OnnxRuntime::Model.new("../onnx/unet/model.onnx", providers:)
channels_num = unet.inputs.detect{ |e| e[:name] == "sample" }[:shape][1]
generator = Torch::Generator.new.manual_seed(0) # Seed generator to create the initial latent noise
latents = Torch.randn([batch_size, channels_num, height / 8, width / 8], generator:, device:) # Shape: 1x4x64x64

# vae_encoder = OnnxRuntime::Model.new("../onnx/vae_encoder/model.onnx", providers:)
# vae_decoder = OnnxRuntime::Model.new("./onnx/vae_decoder/model.onnx", providers:)
#

num_inference_steps = 25 # denoising steps
guidance_scale = 7.5 # classifier-free guidance

debugger
