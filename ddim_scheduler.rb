require "torch-rb"

# Example:
# scheduler = DDIMScheduler.new
# puts "Alphas: #{scheduler.alphas[0..9]}" # Prints the first 10 alphas
# puts "Timesteps: #{scheduler.timesteps[0..9]}" # Prints the first 10 timesteps

class DDIMScheduler
  attr_reader :beta_schedule, :beta_start, :beta_end,
    :num_train_timesteps, :timestep_spacing,
    :prediction_type, :clip_sample, :set_alpha_to_one,
    :skip_prk_steps, :steps_offset, :trained_betas, :alphas, :alphas_cumprod,
    :init_noise_sigma

  attr_accessor :num_inference_steps

  def initialize(
    beta_schedule: "scaled_linear",
    beta_start: 0.00085,
    beta_end: 0.012,
    num_train_timesteps: 1000,
    num_inference_steps: 25,
    timestep_spacing: "leading",
    prediction_type: "epsilon",
    clip_sample: false,
    set_alpha_to_one: false,
    skip_prk_steps: true,
    steps_offset: 0,
    trained_betas: nil
  )
    @beta_start = beta_start
    @beta_end = beta_end
    @beta_schedule = beta_schedule
    @num_train_timesteps = num_train_timesteps
    @num_inference_steps = num_inference_steps
    @timestep_spacing = timestep_spacing
    @prediction_type = prediction_type
    @clip_sample = clip_sample
    @set_alpha_to_one = set_alpha_to_one
    @skip_prk_steps = skip_prk_steps
    @steps_offset = steps_offset
    @trained_betas = trained_betas
    @init_noise_sigma = 1.0

    initialize_betas_and_alphas
    initialize_alphas_cumprod
  end

  def timesteps
    case timestep_spacing
    when "linspace"
      linspace_timesteps
    else
      leading_timesteps
    end
  end

  def scale_model_input(input, timestep: 0)
    input
  end

  def step(model_output, timestep, sample, eta: 0.0)
    # 获取当前时间步的参数
    alpha_t = alphas_cumprod[timestep]
    alpha_t_prev = timestep > 0 ? alphas_cumprod[timestep - 1] : Torch.tensor(1.0)

    beta_t = Torch.tensor(1.0) - alpha_t
    sigma_t = Torch.tensor(eta) * Torch.sqrt(beta_t) # DDIM 的噪声项

    # 计算去噪后的样本 x_0
    pred_original_sample = (sample - Torch.sqrt(beta_t) * model_output) / Torch.sqrt(alpha_t)

    # 计算下一个时间步的样本
    pred_sample_direction = Torch.sqrt(Torch.tensor(1.0) - alpha_t_prev - sigma_t ** 2) * model_output
    x_prev = Torch.sqrt(alpha_t_prev) * pred_original_sample + pred_sample_direction

    # 如果 eta > 0，加入随机噪声
    if eta > 0.0
      noise = Torch.randn(sample.size) # Shape: 1x4x64x64
      x_prev += Torch.tensor(eta) * Torch.sqrt(Torch.tensor(1.0) - alpha_t_prev) * noise
    end

    { prev_sample: x_prev }
  end

  private

    def initialize_betas_and_alphas
      if @trained_betas.nil?
        case @beta_schedule
        when "scaled_linear"
          @betas = scaled_linear_beta_schedule(@beta_start, @beta_end, @num_train_timesteps)
        else
          raise "Unsupported beta_schedule: #{@beta_schedule}"
        end
      else
        @betas = @trained_betas
      end

      @alphas = @betas
        .map { |beta| 1.0 - beta }
        .tap { |a| a[-1] = 1.0 if @set_alpha_to_one }
    end

    def scaled_linear_beta_schedule(beta_start, beta_end, num_train_timesteps)
      (0...num_train_timesteps).map do |t|
        beta_start + t * (beta_end - beta_start) / (num_train_timesteps - 1)
      end
    end

    def initialize_alphas_cumprod
      @alphas_cumprod = Torch.cumprod(Torch.tensor(alphas), dim: 0)
    end

    def leading_timesteps
      step = num_train_timesteps / num_inference_steps
      Torch.arange(0, num_train_timesteps, step)
        .round
        .flip(dims: [0])
        .then { |t| t + steps_offset.to_i }
        .map(&:to_i)
    end

    def linspace_timesteps
      Torch.linspace(0, num_train_timesteps.pred, num_inference_steps)
        .round
        .flip(dims: [0])
        .then { |t| t + steps_offset.to_i }
        .clip(0, num_train_timesteps.pred)
        .map(&:to_i)
    end

    def native_linspace_timesteps
      (0...num_inference_steps)
        .to_a
        .tap { |ts| ts.shift(@steps_offset) if @steps_offset > 0 }
        .map { |t| ((num_train_timesteps - 1).to_f / (num_inference_steps - 1) * t).round }
        .reverse
    end
end
