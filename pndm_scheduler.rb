require "torch-rb"

class PNDMScheduler
  attr_reader :beta_schedule, :beta_start, :beta_end, :betas,
    :alphas, :alphas_cumprod,
    :num_train_timesteps, :timestep_spacing,
    :prediction_type,
    :clip_sample, :set_alpha_to_one,
    :skip_prk_steps, :steps_offset, :trained_betas,
    :init_noise_sigma, :final_alpha_cumprod, :prk_timesteps

  attr_accessor :num_inference_steps, :ets, :counter, :cur_sample, :cur_model_output

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
    skip_prk_steps: false,
    steps_offset: 0,
    trained_betas: nil,
    ets: [],
    counter: 0
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
    @prk_timesteps = []
    @cur_model_output = 0.0
    @cur_sample = nil
    @ets = ets
    @counter = counter

    # 初始化 betas 和 alphas_cumprod
    initialize_betas_and_alphas
    compute_alphas_cumprod

    @final_alpha_cumprod = set_alpha_to_one ? torch.tensor(1.0) : alphas_cumprod[0]
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
    if (counter < prk_timesteps.length) && !skip_prk_steps
      step_prk(model_output, timestep, sample)
    else
      step_plms(model_output, timestep, sample)
    end
  end

  def step_prk(model_output, timestep, sample)
    diff_to_prev = counter % 2 ? 0 : num_train_timesteps / num_inference_steps / 2
    prev_timestep = timestep - diff_to_prev
    timestep = prk_timesteps[counter / 4 * 4]

    if counter % 4 == 0
      @cur_model_output += model_output / 6.0
      @ets.append(model_output)
      @cur_sample = sample
    elsif (counter - 1) % 4 == 0
      @cur_model_output += model_output / 3.0
    elsif (counter - 2) % 4 == 0
      @cur_model_output += model_output / 3.0
    elsif (counter - 3) % 4 == 0
      model_output = cur_model_output + model_output / 6.0
      @cur_model_output = 0
    end

    cur_sample = cur_sample.nil? ? sample : cur_sample
    prev_sample = get_prev_sample(cur_sample, timestep, prev_timestep, model_output)
    @counter += 1

    { prev_sample: }
  end

  def step_plms(model_output, timestep, sample)
    prev_timestep = timestep - num_train_timesteps / num_inference_steps

    if counter != 1
      @ets = @ets.last(3)
      @ets.append(model_output)
    else
        prev_timestep = timestep
        timestep = timestep + num_train_timesteps / num_inference_steps
    end

    if (ets.length == 1) && (counter == 0)
        model_output = model_output
        @cur_sample = sample
    elsif (ets.length == 1) && (counter == 1)
      model_output = (model_output + ets[-1]) / 2.0
      sample = cur_sample
      @cur_sample = nil
    elsif ets.length == 2
      model_output = (3 * ets[-1] - ets[-2]) / 2.0
    elsif ets.length == 3
      model_output = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12.0
    else
      model_output = (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24.0
    end

    prev_sample = get_prev_sample(sample, timestep, prev_timestep, model_output)
    @counter += 1

    puts prev_sample
    { prev_sample: }
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

    def get_prev_sample(sample, timestep, prev_timestep, model_output)
      alpha_prod_t = alphas_cumprod[timestep]
      alpha_prod_t_prev = prev_timestep >= 0 ? alphas_cumprod[prev_timestep] : final_alpha_cumprod
      beta_prod_t = 1 - alpha_prod_t
      beta_prod_t_prev = 1 - alpha_prod_t_prev

      if prediction_type == "v_prediction"
        model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
      elsif prediction_type != "epsilon"
        raise "prediction_type must be one of epsilon or v_prediction"
      end

      sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
      model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** 0.5 + (alpha_prod_t * beta_prod_t * alpha_prod_t_prev) ** 0.5

      prev_sample = (
          sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
      )

      prev_sample
    end

    def _add_noise(original_sample, noise, timestep)
      alpha_t = Math.cos(timestep * Math::PI / 2)
      (alpha_t * original_sample) + ((1 - alpha_t) * noise)
    end
end
