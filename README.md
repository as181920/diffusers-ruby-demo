# Diffusers ruby demo

Run huggingface diffusers model by Ruby

## Run

```bash
optimum-cli export onnx --model stable-diffusion-v1-5/stable-diffusion-v1-5 onnx
git clone git@github.com:as181920/diffusers-ruby-demo.git
ruby demo.rb # => output-rb.png
```

## DEBUG

Check calc differences to python code step by step:

```bash
# git clone git@github.com:huggingface/diffusers.git local_diffusers
# change import comments to use local diffusers
python demo.py

LOG_LEVEL=debug ruby demo.rb
```

## Thanks

[Andrew Kane](https://github.com/ankane)
