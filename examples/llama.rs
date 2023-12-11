use anyhow::{bail, Error as E, Result};
use clap::Parser;

use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use tokenizers::Tokenizer;

use candle_transformers::models::llama as model;
use model::{Llama, LlamaConfig};

const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

pub fn device() -> Result<Device> {
    if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    #[arg(long)]
    model_id: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
    #[arg(long)]
    local_weights: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Get device and set dtype
    let device = device()?;
    let dtype = DType::F16;

    // Load the model from huggingface.co
    let api = Api::new()?;
    let model_id = args
        .model_id
        .unwrap_or_else(|| "NousResearch/Llama-2-7b-chat-hf".to_string());
    println!("loading the model weights from {model_id}");
    let revision = "main".to_string();
    let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    let tokenizer_filename = match &args.local_weights {
        Some(path) => (path.to_owned() + "tokenizer.json").into(),
        _ => api.get("tokenizer.json")?,
    };

    let config_filename = match &args.local_weights {
        Some(path) => (path.to_owned() + "config.json").into(),
        _ => api.get("config.json")?,
    };
    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let config = config.into_config(true);

    let mut filenames = vec![];
    for rfilename in [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ] {
        match &args.local_weights {
            Some(path) => {
                filenames.push((path.to_owned() + rfilename).into());
            }
            _ => {
                let filename = api.get(rfilename)?;
                filenames.push(filename);
            }
        };
    }

    // Build and load the model with KV cache
    println!("building the model");
    let cache = model::Cache::new(true, dtype, &config, &device)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let llama = Llama::load(vb, &cache, &config)?;

    // Load the tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let eos_token_id = tokenizer.token_to_id(EOS_TOKEN);

    // Inference loop
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    println!("starting the inference loop");
    print!("{prompt}");
    let seed = 42;
    let mut logits_processor = LogitsProcessor::new(seed, args.temperature, args.top_p);
    let start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    for index in 0..args.sample_len {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, index_pos)?;
        let logits = logits.squeeze(0)?;
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        // Extracting the last token as a string is complicated, here we just apply some simple
        // heuristics as it seems to work well enough for this example. See the following for more
        // details:
        // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
        if let Some(text) = tokenizer.id_to_token(next_token) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            print!("{text}");
            std::io::stdout().flush()?;
        }
        if Some(next_token) == eos_token_id {
            break;
        }
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        token_generated as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
