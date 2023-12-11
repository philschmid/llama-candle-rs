# Candle example on how to run Llama on NVIDIA A10G GPU

This example shows how to run Llama on NVIDIA A10G GPU. The example is based on the [Candle](https://github.com/huggingface/candle). Candle is a minimalist ML framework for Rust with a focus on performance (including GPU support) and ease of use.

This example walks through the following steps:
1. Create a new Rust project
2. Install Candle for GPU support
3. Install Rust dependencies
3. Run Llama on GPU using Candle
4. Run GGUF (llama.cpp) on GPU using Candle

Before we get started make sure you have `cargo` and `rustup` installed. If not, follow the instructions [here](https://www.rust-lang.org/tools/install).

## Create a new Rust project

The first step is to create a new Rust project. To do so, run the following command:

```bash
cargo new --bin llama-candle
```

This will create a new Rust project with the name `llama-candle`. The project will contain a `src` folder with a `main.rs` file. The `main.rs` file contains a simple "Hello World" example. You can run the example by running the following command:

```bash
cd llama-candle
cargo run
```

This will print "Hello, world!" to the console. Perfect! Now let's install Candle for GPU support.


## Install Candle for GPU support

We will work along side the installation instructions from the [Candle](https://huggingface.github.io/candle/guide/installation.html). First lets check we have access to a GPU. To do so, run the following command:

```bash
nvcc --version # 11.7
nvidia-smi --query-gpu=compute_cap --format=csv # 8.6
```

To add candle and all subpackages needed to run Llama using Flash Attention, add the following line to your `Cargo.toml` file under `[dependencies]`:

_Note: If you run on mac (m1 or newer) change the features of `candle-core` from "cuda" to "metal" and for `candle-nn`, `candle-transformers` from "cuda" to "accelerate"._

```toml
# candle specific dependencies
candle-core = { git = "https://github.com/huggingface/candle.git", features = ["cuda"], version = "0.3.1" }
candle-nn = { git = "https://github.com/huggingface/candle.git", features = ["cuda"], version = "0.3.1" }
candle-transformers = { git = "https://github.com/huggingface/candle.git", features = ["cuda","flash-attn"], version = "0.3.1" }
tokenizers = "0.13.4"
hf-hub = "0.3.2"
# general dependencies
serde = { version = "1.0.190", features = ["serde_derive"] }
clap = { version = "4.4.7", features = ["derive"] }
anyhow = "1.0.75"
serde_json = "1.0.108"
```

No we can build and install the dependencies by running the following command:

```bash
cargo build
```

_Note: Depending on your system, this might take a few minutes (10-15), especially since we are building the dependencies from source for flash attention._

If you get an error open an issue or check the [Common Errors](https://github.com/huggingface/candle/tree/main#common-errors) page.

## 3. Run Llama on GPU using Candle

We have everything we need to run Llama on GPU using Candle. Now, we need our modelling, tokenizer and generation code. To make it easier I added a adjusted example from the candle repository to this repository. You can find the code in the [examples](examples/) folder. 

We can download the examples and replace our main.rs file with the example by running the following command:

```bash
wget -O src/main.rs http://example.com/file.txt
```

Now, lets test it with the following command: 
```bash
cargo run -- model-id NousResearch/Llama-2-7b-chat-hf
```

_Note: The code only works with 7B models since we "hardcoded" the modelling file names._

Awesome it works, but its a bit slow and we didn't define a prompt or any parameter. Lets change this with the following command:

_Note: the `release` flag will build the code in release mode, which will make it faster, but also takes some time again._

```bash
cargo run --release -- --prompt 'Write helloworld code in Rust' --sample-len 150
```

## 4. Run GGUF (llama.cpp) on GPU using Candle


# Todos

* [ ] clean code
  * [ ] no local files
  * [ ] helper method to load the model
  * [ ] helper method for generation
* [ ] add quantization example