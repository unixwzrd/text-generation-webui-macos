# Text generation web UI - Modified for macOS and Apple Silicon 2024-05-10 Edition

## This is the original oobabooga text generation webui modified to run on macOS 

This is a dev release, documentation under re-work, there will probably be changes before final release.

This is a development version and I have not added many changes I had planned. Please ||feel|| free to use at your own risk as there may be bugs not yet found.

Items Added to this version.
 * Added ElevenLabs extension back

Items working and tested on macOS
 * More support for Apple Silicon M1/M2/M3 processors
 * Working with new llama-cpp-python 0.1.81
 * Works with LLaMa2 Models
        * The pip recompile of llama-cpp-python has changed.
    
Removed from this
 * Tried to continue what was already started in removing FlexGEN from the repo
 * Removed Docker - if someone wants to help maintain for macOS, let me know
 * Slowly removing information on CUDA as it is not relevant to macOS

  **Updated Installation Instructions** for libraries in the [oobabooga-macOS Quickstart](https://github.com/unixwzrd/oobabooga-macOS/blob/main/macOS_Apple_Silicon_QuickStart.m1) and the longer [Building Apple Silicon Support](https://github.com/unixwzrd/oobabooga-macOS/blob/main/macOS-Install.md)

If you want the most recent version, from the oobabooga repository, go here: [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)

Otherwise, use these instructions I have on putting together the macOS Python environment. These instructions are not only useful for setting up oobabooga, but also for anyone working in data analytics, machine learning, deep learning, scientific computing, and other areas that can benefit from an optimized Python GPU environment on Apple Silicon.

* [Building Apple Silicon Support for oobabooga text-generation-webui](https://github.com/unixwzrd/oobabooga-macOS/blob/main/macOS-Install.md)
* [oobabooga macOS Apple Silicon Quick Start for the Impatient](https://github.com/unixwzrd/oobabooga-macOS/blob/main/macOS_Apple_Silicon_QuickStart.m1)

I will be updating this README file with new information specifically regarding macOS and Apple Silicon.

Maintaining and improving support for macOS and Apple Silicon in this project has required significant research, debugging, and development effort. If you find my contributions helpful and want to show your appreciation, you can Buy Me a Coffee, sponsor this project, or consider me for job opportunities.

While the focus of this branch is to enhance macOS and Apple Silicon support, I aim to maintain compatibility with Linux and POSIX operating systems. Contributions and feedback related to Linux compatibility are always welcome.

Anyone who would like to assist with supporting Apple Silicon, let me know. There is much to do and I can only do so much by myself.

## All the features of the UI will run on macOS and have been tested on the following configurations, using only llama.cpp

There are CUDA issues to work out, and I'd like to find a better way around this, but wanted to get this out as soon as I could and continue to work on the other in thw background.

|   Hardware                       | Memory | macOS Name | Version |
|----------------------------------|--------|------------|---------|
| MacBook Pro 16" M2 Max Processor |  96GB  |  Sonoma    | 14.5    |

- [Text generation web UI - Modified for macOS and Apple Silicon 2024-05-10 Edition](#text-generation-web-ui---modified-for-macos-and-apple-silicon-2024-05-10-edition)
  - [This is the original oobabooga text generation webui modified to run on macOS](#this-is-the-original-oobabooga-text-generation-webui-modified-to-run-on-macos)
  - [All the features of the UI will run on macOS and have been tested on the following configurations, using only llama.cpp](#all-the-features-of-the-ui-will-run-on-macos-and-have-been-tested-on-the-following-configurations-using-only-llamacpp)
  - [Features](#features)
  - [Installation process](#installation-process)
    - [Install Miniconda](#install-miniconda)
      - [Download the miniconda installer](#download-the-miniconda-installer)
  - [Startup Options](#startup-options)
      - [Basic settings](#basic-settings)
      - [Model loader](#model-loader)
      - [Accelerate/transformers](#acceleratetransformers)
      - [bitsandbytes 4-bit](#bitsandbytes-4-bit)
      - [llama.cpp](#llamacpp)
      - [ExLlamav2](#exllamav2)
      - [AutoGPTQ](#autogptq)
      - [GPTQ-for-LLaMa](#gptq-for-llama)
      - [HQQ](#hqq)
      - [DeepSpeed](#deepspeed)
      - [RoPE (for llama.cpp, ExLlamaV2, and transformers)](#rope-for-llamacpp-exllamav2-and-transformers)
      - [Gradio](#gradio)
      - [API](#api)
      - [Multimodal](#multimodal)
  - [Documentation](#documentation)
  - [Downloading models](#downloading-models)
  - [Contributing](#contributing)
  - [Acknowledgments](#acknowledgments)


A Gradio web UI for Large Language Models, running on macOS

The goal of this project is to bring oobabooga to macOS.

oobabooga's goal is to become the [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) of text generation.

|![Image1](https://github.com/oobabooga/screenshots/raw/main/print_instruct.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/print_chat.png) |
|:---:|:---:|
|![Image1](https://github.com/oobabooga/screenshots/raw/main/print_default.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/print_parameters.png) |

## Features

* 3 interface modes: default (two columns), notebook, and chat.
* Only [llama.cpp](https://github.com/ggerganov/llama.cpp) for now.
  * Multiple model backends: [Transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp) (through [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)), [ExLlamaV2](https://github.com/turboderp/exllamav2), [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp).
* Dropdown menu for quickly switching between different models.
* Large number of extensions (built-in and user-contributed), including Coqui TTS for realistic voice outputs, Whisper STT for voice inputs, translation, [multimodal pipelines](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/multimodal), vector databases, Stable Diffusion integration, and a lot more. See [the wiki](https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions) and [the extensions directory](https://github.com/oobabooga/text-generation-webui-extensions) for details.
* [Chat with custom characters](https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#character).
* Precise chat templates for instruction-following models, including Llama-2-chat, Alpaca, Vicuna, Mistral.
* LoRA: train new LoRAs with your own data, load/unload LoRAs on the fly for generation.
* Transformers library integration: load models in 4-bit or 8-bit precision through bitsandbytes, use llama.cpp with transformers samplers (`llamacpp_HF` loader), CPU inference in 32-bit precision using PyTorch.
* OpenAI-compatible API server with Chat and Completions endpoints -- see the [examples](https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API#examples).


## Installation process

### Install Miniconda

#### Download the miniconda installer

```bash
curl  https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh


#### Run the installer in non-destructive mode in order to preserve any existing installation.
sh miniconda.sh -b -u

. "${HOME}/miniconda3/bin/activate"

conda init $(basename "${SHELL}")
conda update -n base -c defaults conda -y
 
#### Get a new login shell

exec bash -l
conda create -n llama-env python=3.10 -y
conda activate llama-env

### Build and install CMake

#### Clone the CMake repository, build, and install CMake

git clone https://github.com/Kitware/CMake.git
cd CMake
git checkout tags/v3.29.3
mkdir build
cd build

#### This will configure the installation of cmake to be in your home directory under local, rather than /usr/local

../bootstrap --prefix=${HOME}/local
make -j
make -j test
make install

#### Be sure to add ${HOME}/local/bin to your path  **Add to your .profile, .bashrc, etc...**

export PATH=${HOME}/local/bin:${PATH}

#### Verify the installation

which cmake       # Should say $HOME/local/bin
cmake --version

### Get my oobabooga and checkout macOS-test branch

git clone https://github.com/unixwzrd/text-generation-webui-macos.git textgen-macOS
cd textgen-macOS
git checkout macOS-dev
pip install -r requirements.txt

#### llamacpp-python

export CMAKE_ARGS="-DLLAMA_METAL=on"
export FORCE_CMAKE=1
export PATH=/usr/local/bin:$PATH  # Ensure the correct cmake is used
pip install llama-cpp-python --force-reinstall --no-cache --no-binary :all: --compile --no-deps --no-build-isolation

#### Pip PyTorch install from daily build
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu --force-reinstall --no-deps

#### NumPy Rebuild with Pip
export CFLAGS="-I/System/Library/Frameworks/vecLib.framework/Headers -Wl,-framework -Wl,Accelerate -framework Accelerate"; pip install numpy --force-reinstall --no-deps --no-cache --no-binary :all: --no-build-isolation --compile -Csetup-args=-Dblas=accelerate -Csetup-args=-Dlapack=accelerate -Csetup-args=-Duse-ilp64=true

#### CTransformers
export CFLAGS="-I/System/Library/Frameworks/vecLib.framework/Headers -Wl,-framework -Wl,Accelerate -framework Accelerate"; export CT_METAL=1; pip install ctransformers --no-binary :all: --no-deps --no-build-isolation --compile --force-reinstall -v
unset CMAKE_ARGS FORCE_CMAKE CFLAGS CT_METAL
```

## Startup Options
<details>
<summary>
<b>List of command-line flags</b>
</summary>

#### Basic settings

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | show this help message and exit |
| `--multi-user`                             | Multi-user mode. Chat histories are not saved or automatically loaded. WARNING: this is likely not safe for sharing publicly. |
| `--character CHARACTER`                    | The name of the character to load in chat mode by default. |
| `--model MODEL`                            | Name of the model to load by default. |
| `--lora LORA [LORA ...]`                   | The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces. |
| `--model-dir MODEL_DIR`                    | Path to directory with all the models. |
| `--lora-dir LORA_DIR`                      | Path to directory with all the loras. |
| `--model-menu`                             | Show a model menu in the terminal when the web UI is first launched. |
| `--settings SETTINGS_FILE`                 | Load the default interface settings from this yaml file. See `settings-template.yaml` for an example. If you create a file called `settings.yaml`, this file will be loaded by default without the need to use the `--settings` flag. |
| `--extensions EXTENSIONS [EXTENSIONS ...]` | The list of extensions to load. If you want to load more than one extension, write the names separated by spaces. |
| `--verbose`                                | Print the prompts to the terminal. |
| `--chat-buttons`                           | Show buttons on the chat tab instead of a hover menu. |

#### Model loader

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `--loader LOADER`                          | Choose the model loader manually, otherwise, it will get autodetected. Valid options: Transformers, llama.cpp, llamacpp_HF, ExLlamav2_HF, ExLlamav2, AutoGPTQ, AutoAWQ, GPTQ-for-LLaMa, QuIP#. |

#### Accelerate/transformers

| Flag                                        | Description |
|---------------------------------------------|-------------|
| `--cpu`                                     | Use the CPU to generate text. Warning: Training on CPU is extremely slow. |
| `--auto-devices`                            | Automatically split the model across the available GPU(s) and CPU. |
|  `--gpu-memory GPU_MEMORY [GPU_MEMORY ...]` | Maximum GPU memory in GiB to be allocated per GPU. Example: --gpu-memory 10 for a single GPU, --gpu-memory 10 5 for two GPUs. You can also set values in MiB like --gpu-memory 3500MiB. |
| `--cpu-memory CPU_MEMORY`                   | Maximum CPU memory in GiB to allocate for offloaded weights. Same as above. |
| `--disk`                                    | If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk. |
| `--disk-cache-dir DISK_CACHE_DIR`           | Directory to save the disk cache to. Defaults to "cache". |
| `--load-in-8bit`                            | Load the model with 8-bit precision (using bitsandbytes). |
| `--bf16`                                    | Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU. |
| `--no-cache`                                | Set `use_cache` to `False` while generating text. This reduces VRAM usage slightly, but it comes at a performance cost. |
| `--trust-remote-code`                       | Set `trust_remote_code=True` while loading the model. Necessary for some models. |
| `--no_use_fast`                             | Set use_fast=False while loading the tokenizer (it's True by default). Use this if you have any problems related to use_fast. |
| `--use_flash_attention_2`                   | Set use_flash_attention_2=True while loading the model. |

#### bitsandbytes 4-bit

⚠️  Requires minimum compute of 7.0 on Windows at the moment.

| Flag                                        | Description |
|---------------------------------------------|-------------|
| `--load-in-4bit`                            | Load the model with 4-bit precision (using bitsandbytes). |
| `--use_double_quant`                        | use_double_quant for 4-bit. |
| `--compute_dtype COMPUTE_DTYPE`             | compute dtype for 4-bit. Valid options: bfloat16, float16, float32. |
| `--quant_type QUANT_TYPE`                   | quant_type for 4-bit. Valid options: nf4, fp4. |

#### llama.cpp

| Flag        | Description |
|-------------|-------------|
| `--tensorcores`  | Use llama-cpp-python compiled with tensor cores support. This increases performance on RTX cards. NVIDIA only. |
| `--flash-attn`   | Use flash-attention. |
| `--n_ctx N_CTX` | Size of the prompt context. |
| `--threads` | Number of threads to use. |
| `--threads-batch THREADS_BATCH` | Number of threads to use for batches/prompt processing. |
| `--no_mul_mat_q` | Disable the mulmat kernels. |
| `--n_batch` | Maximum number of prompt tokens to batch together when calling llama_eval. |
| `--no-mmap`   | Prevent mmap from being used. |
| `--mlock`     | Force the system to keep the model in RAM. |
| `--n-gpu-layers N_GPU_LAYERS` | Number of layers to offload to the GPU. |
| `--tensor_split TENSOR_SPLIT`       | Split the model across multiple GPUs. Comma-separated list of proportions. Example: 18,17. |
| `--numa`      | Activate NUMA task allocation for llama.cpp. |
| `--logits_all`| Needs to be set for perplexity evaluation to work. Otherwise, ignore it, as it makes prompt processing slower. |
| `--no_offload_kqv` | Do not offload the K, Q, V to the GPU. This saves VRAM but reduces the performance. |
| `--cache-capacity CACHE_CAPACITY`   | Maximum cache capacity (llama-cpp-python). Examples: 2000MiB, 2GiB. When provided without units, bytes will be assumed. |
| `--row_split`                               | Split the model by rows across GPUs. This may improve multi-gpu performance. |
| `--streaming-llm`                           | Activate StreamingLLM to avoid re-evaluating the entire prompt when old messages are removed. |
| `--attention-sink-size ATTENTION_SINK_SIZE` | StreamingLLM: number of sink tokens. Only used if the trimmed prompt doesn't share a prefix with the old prompt. |

#### ExLlamav2

| Flag             | Description |
|------------------|-------------|
|`--gpu-split`     | Comma-separated list of VRAM (in GB) to use per GPU device for model layers. Example: 20,7,7. |
|`--max_seq_len MAX_SEQ_LEN`           | Maximum sequence length. |
|`--cfg-cache`                         | ExLlamav2_HF: Create an additional cache for CFG negative prompts. Necessary to use CFG with that loader. |
|`--no_flash_attn`                     | Force flash-attention to not be used. |
|`--cache_8bit`                        | Use 8-bit cache to save VRAM. |
|`--cache_4bit`                        | Use Q4 cache to save VRAM. |
|`--num_experts_per_token NUM_EXPERTS_PER_TOKEN` |  Number of experts to use for generation. Applies to MoE models like Mixtral. |

#### AutoGPTQ

| Flag             | Description |
|------------------|-------------|
| `--triton`                     | Use triton. |
| `--no_inject_fused_attention`  | Disable the use of fused attention, which will use less VRAM at the cost of slower inference. |
| `--no_inject_fused_mlp`        | Triton mode only: disable the use of fused MLP, which will use less VRAM at the cost of slower inference. |
| `--no_use_cuda_fp16`           | This can make models faster on some systems. |
| `--desc_act`                   | For models that don't have a quantize_config.json, this parameter is used to define whether to set desc_act or not in BaseQuantizeConfig. |
| `--disable_exllama`            | Disable ExLlama kernel, which can improve inference speed on some systems. |
| `--disable_exllamav2`          | Disable ExLlamav2 kernel. |

#### GPTQ-for-LLaMa

| Flag                      | Description |
|---------------------------|-------------|
| `--wbits WBITS`           | Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported. |
| `--model_type MODEL_TYPE` | Model type of pre-quantized model. Currently LLaMA, OPT, and GPT-J are supported. |
| `--groupsize GROUPSIZE`   | Group size. |
| `--pre_layer PRE_LAYER [PRE_LAYER ...]`  | The number of layers to allocate to the GPU. Setting this parameter enables CPU offloading for 4-bit models. For multi-gpu, write the numbers separated by spaces, eg `--pre_layer 30 60`. |
| `--checkpoint CHECKPOINT` | The path to the quantized checkpoint file. If not specified, it will be automatically detected. |
| `--monkey-patch`          | Apply the monkey patch for using LoRAs with quantized models. |

#### HQQ

| Flag        | Description |
|-------------|-------------|
| `--hqq-backend` | Backend for the HQQ loader. Valid options: PYTORCH, PYTORCH_COMPILE, ATEN. |

#### DeepSpeed

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--deepspeed`                         | Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration. |
| `--nvme-offload-dir NVME_OFFLOAD_DIR` | DeepSpeed: Directory to use for ZeRO-3 NVME offloading. |
| `--local_rank LOCAL_RANK`             | DeepSpeed: Optional argument for distributed setups. |

#### RoPE (for llama.cpp, ExLlamaV2, and transformers)

| Flag             | Description |
|------------------|-------------|
| `--alpha_value ALPHA_VALUE`           | Positional embeddings alpha factor for NTK RoPE scaling. Use either this or `compress_pos_emb`, not both. |
| `--rope_freq_base ROPE_FREQ_BASE`     | If greater than 0, will be used instead of alpha_value. Those two are related by `rope_freq_base = 10000 * alpha_value ^ (64 / 63)`. |
| `--compress_pos_emb COMPRESS_POS_EMB` | Positional embeddings compression factor. Should be set to `(context length) / (model's original context length)`. Equal to `1/rope_freq_scale`. |

#### Gradio

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--listen`                            | Make the web UI reachable from your local network. |
| `--listen-port LISTEN_PORT`           | The listening port that the server will use. |
| `--listen-host LISTEN_HOST`           | The hostname that the server will use. |
| `--share`                             | Create a public URL. This is useful for running the web UI on Google Colab or similar. |
| `--auto-launch`                       | Open the web UI in the default browser upon launch. |
| `--gradio-auth USER:PWD`              | Set Gradio authentication password in the format "username:password". Multiple credentials can also be supplied with "u1:p1,u2:p2,u3:p3". |
| `--gradio-auth-path GRADIO_AUTH_PATH` | Set the Gradio authentication file path. The file should contain one or more user:password pairs in the same format as above. |
| `--ssl-keyfile SSL_KEYFILE`           | The path to the SSL certificate key file. |
| `--ssl-certfile SSL_CERTFILE`         | The path to the SSL certificate cert file. |

#### API

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--api`                               | Enable the API extension. |
| `--public-api`                        | Create a public URL for the API using Cloudfare. |
| `--public-api-id PUBLIC_API_ID`       | Tunnel ID for named Cloudflare Tunnel. Use together with public-api option. |
| `--api-port API_PORT`                 | The listening port for the API. |
| `--api-key API_KEY`                   | API authentication key. |
| `--admin-key ADMIN_KEY`               | API authentication key for admin tasks like loading and unloading models. If not set, will be the same as --api-key. |
| `--nowebui`                           | Do not launch the Gradio UI. Useful for launching the API in standalone mode. |

#### Multimodal

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--multimodal-pipeline PIPELINE`      | The multimodal pipeline to use. Examples: `llava-7b`, `llava-13b`. |

</details>

## Documentation

https://github.com/oobabooga/text-generation-webui/wiki

## Downloading models

Models should be placed in the folder `text-generation-webui/models`. They are usually downloaded from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads).

* GGUF models are a single file and should be placed directly into `models`. Example:

```
text-generation-webui
└── models
    └── llama-2-13b-chat.Q4_K_M.gguf
```

* The remaining model types (like 16-bit transformers models and GPTQ models) are made of several files and must be placed in a subfolder. Example:

```
text-generation-webui
├── models
│   ├── lmsys_vicuna-33b-v1.3
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── pytorch_model-00001-of-00007.bin
│   │   ├── pytorch_model-00002-of-00007.bin
│   │   ├── pytorch_model-00003-of-00007.bin
│   │   ├── pytorch_model-00004-of-00007.bin
│   │   ├── pytorch_model-00005-of-00007.bin
│   │   ├── pytorch_model-00006-of-00007.bin
│   │   ├── pytorch_model-00007-of-00007.bin
│   │   ├── pytorch_model.bin.index.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.model
```

In both cases, you can use the "Model" tab of the UI to download the model from Hugging Face automatically. It is also possible to download it via the command-line with 

```
python download-model.py organization/model
```

Run `python download-model.py --help` to see all the options.

## Contributing

Get in contact or post to the GutHub Discussions

## Acknowledgments

The entire oobabooga team.
