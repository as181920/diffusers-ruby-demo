require "torch-rb"

class PNDMScheduler
  attr_reader :beta_schedule, :beta_start, :beta_end, :betas,
    :alphas, :alphas_cumprod,
    :num_train_timesteps, :timestep_spacing,
    :prediction_type,
    :clip_sample, :set_alpha_to_one,
    :skip_prk_steps, :steps_offset, :trained_betas,
    :init_noise_sigma

  attr_accessor :num_inference_steps, :ets, :counter

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
    trained_betas: nil,
    ets: []
  )
    # 从配置初始化参数
    @beta_schedule = beta_schedule
    @beta_start = beta_start
    @beta_end = beta_end
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
    @ets = ets
    @counter = 0

    # 初始化 betas 和 alphas_cumprod
    initialize_betas_and_alphas
    compute_alphas_cumprod
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

  def step(model_output, timestep, sample)
    if ets.size < 4
      step_prk(model_output, timestep, sample)
    else
      step_plms(model_output, timestep, sample)
    end
  end

  def step_prk(model_output, timestep, sample)
    @ets << model_output
    alpha_t = Math.cos(timestep * Math::PI / 2)
    beta_t = 1 - alpha_t

    updated_sample = sample - (Torch.tensor(beta_t) * model_output)

    if @ets.size == 4 # Once 4 ets are gathered, use PLMS directly in the next step
      @counter += 1
      @ets.shift
    end

    { prev_sample: updated_sample }
  end

  def step_plms(model_output, timestep, sample)
    @ets << model_output
    @ets.shift if @ets.size > 4

    a, b, c, d = [0.5, -1.0, 1.5, -0.5].map { |f| Torch.tensor(f) } # Coefficients for PLMS
    combined_eta = (a * @ets[-4]) + (b * @ets[-3]) + (c * @ets[-2]) + (d * @ets[-1])

    alpha_t = Math.cos(timestep * Math::PI / 2)
    beta_t = 1 - alpha_t

    updated_sample = sample - (Torch.tensor(beta_t) * combined_eta)
    { prev_sample: updated_sample }
  end

  def step_deprecated(model_output, timestep, sample)
    # 当前 alpha_cumprod 和 beta_t
    alpha_cumprod_t = alphas_cumprod[timestep]
    alpha_cumprod_prev = timestep > 0 ? alphas_cumprod[timestep - 1] : Torch.tensor(1.0)
    # beta_t = betas[timestep]

    # 根据 prediction_type 计算 x_0
    case prediction_type
    when "epsilon"
      pred_original_sample = (sample - Torch.sqrt(Torch.tensor(1.0) - alpha_cumprod_t) * model_output) / Torch.sqrt(alpha_cumprod_t)
    else
      raise NotImplementedError, "Prediction type #{prediction_type} not implemented."
    end

    # Clip x_0 if clip_sample is true
    pred_original_sample = pred_original_sample.map { |v| v.clamp(-1.0, 1.0) } if clip_sample

    # 计算下一个时间步的样本
    sample_predicted = Torch.sqrt(alpha_cumprod_prev) * pred_original_sample +
      Torch.sqrt(Torch.tensor(1.0) - alpha_cumprod_prev) * model_output

    { prev_sample: sample_predicted }
  end

  private

    def initialize_betas_and_alphas
      if @trained_betas.nil?
        case beta_schedule
        when "scaled_linear"
          @betas = Torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else
          raise NotImplementedError, "Beta schedule #{beta_schedule} not implemented."
        end
      else
        @betas = @trained_betas
      end

      @alphas = (Torch.tensor(1.0) - betas)
        .tap { |a| a[-1] = Torch.tensor(1.0) if set_alpha_to_one }
    end

    def compute_alphas_cumprod
      @alphas_cumprod = Torch.cumprod(alphas, dim: 0)
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

    def _add_noise(original_sample, noise, timestep)
      alpha_t = Math.cos(timestep * Math::PI / 2)
      (alpha_t * original_sample) + ((1 - alpha_t) * noise)
    end
end
