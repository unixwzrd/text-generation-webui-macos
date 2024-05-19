# Text generation web UI - Modified for macOS and Apple Silicon 2024-05-10 Edition

This is a dev release, documentation under re-work, there will probably be changes before final relese.

This is a development version and I have not added many changes I had planned. Please ||feel|| free to use at your own risk as there may be bugs not yet found.

Items Added to this version.
 * "Stop Server" under the sessions tab. Use with caution if in multi-user, will probably disable this if in multi-user mode, however it offers better shutdown than just killing the process on the server.
 * Added ElenevLabs extension back.

Items working and tested on macOS
 * More support for Apple Silicon M1/M2/M3 processors.
 * Working with new llama-cpp-python 0.1.81
 * Works with LLaMa2 Models
        * There GGML models will need conversion to GGUF format if using llama-cpp-python 0.1.81.
        * Earlier version llama-coo-python still works
        * Have not concluded testing of library dependencies, will have that updated in build instructions for oobagooba-macOS, it will require an older version of llama-cpp-python.
        * Only GGUF files for llama.cpp IF you need this, please create a PR, thanks!
  
Removed from this
 * Tried to continue what was already started in removing FlexGEN from the repo.
 * Removed Docker - if someone wants to help maintain for macOS, let me know.
 * Slowly removing information on CUDA as it is not relevant to macOS.

  **Updated Installation Instructions** for libraries in the [oobabooga-macOS Quickstart][1] and the longer [Building Apple Silicon Support][2]

GGML support is in this release, and has not been extensively tested. From the look of upstream commits, there are some changes which must be made before this will work with Llama2 models.

If you want the most recent version, from the oobabooga repository, go here: [oobabooga/text-generation-webgui][3]

Otherwise, use these instructions I have on putting together the macOS Python environment. These instructions are not only useful for setting up oobabooga, but also for anyone working in data analytics, machine learning, deep learning, scientific computing, and other areas that can benefit from an optimized Python GPU environment on Apple Silicon.

* [Building Apple Silicon Support for oobabooga text-generation-webui][4]
* [oobabooga macOS Apple Silicon Quick Start for the Impatient][5]

I will be updating this README file with new information specifically regarding macOS and Apple Silicon.

I would like to work closely with the oobabooga team and try to implement similar solutions so the web UI can have a similar look and feel.

Maintaining and improving support for macOS and Apple Silicon in this project has required significant research, debugging, and development effort. If you find my contributions helpful and want to show your appreciation, you can Buy Me a Coffee, sponsor this project, or consider me for job opportunities.

While the focus of this branch is to enhance macOS and Apple Silicon support, I aim to maintain compatibility with Linux and POSIX operating systems. Contributions and feedback related to Linux compatibility are always welcome.

Anyone who would like to assist with supporting Apple Silicon, let me know. There is much to do and I can only do so much by myself.

- [Text generation web UI - Modified for macOS and Apple Silicon 2024-05-10 Edition](#text-generation-web-ui---modified-for-macos-and-apple-silicon-2024-05-10-edition)
  - [Features](#features)
  - [How to install](#how-to-install)
    - [One-click-installer](#one-click-installer)
    - [Manual installation using Conda](#manual-installation-using-conda)
      - [0. Install Conda](#0-install-conda)
      - [1. Create a new conda environment](#1-create-a-new-conda-environment)
      - [2. Install Pytorch](#2-install-pytorch)
      - [3. Install the web UI](#3-install-the-web-ui)
    - [Start the web UI](#start-the-web-ui)
        - [AMD GPU on Windows](#amd-gpu-on-windows)
        - [Older NVIDIA GPUs](#older-nvidia-gpus)
        - [Manual install](#manual-install)
    - [Alternative: Docker](#alternative-docker)
    - [Updating the requirements](#updating-the-requirements)
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
  - [Google Colab notebook](#google-colab-notebook)
  - [Contributing](#contributing)
  - [Community](#community)
  - [Acknowledgment](#acknowledgment)


A Gradio web UI for Large Language Models.

Its goal is to become the [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) of text generation.

|![Image1](https://github.com/oobabooga/screenshots/raw/main/print_instruct.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/print_chat.png) |
|:---:|:---:|
|![Image1](https://github.com/oobabooga/screenshots/raw/main/print_default.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/print_parameters.png) |

## Features

* 3 interface modes: default (two columns), notebook, and chat.
* Multiple model backends: [Transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp) (through [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)), [ExLlamaV2](https://github.com/turboderp/exllamav2), [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp).
* Dropdown menu for quickly switching between different models.
* Large number of extensions (built-in and user-contributed), including Coqui TTS for realistic voice outputs, Whisper STT for voice inputs, translation, [multimodal pipelines](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/multimodal), vector databases, Stable Diffusion integration, and a lot more. See [the wiki](https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions) and [the extensions directory](https://github.com/oobabooga/text-generation-webui-extensions) for details.
* [Chat with custom characters](https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#character).
* Precise chat templates for instruction-following models, including Llama-2-chat, Alpaca, Vicuna, Mistral.
* LoRA: train new LoRAs with your own data, load/unload LoRAs on the fly for generation.
* Transformers library integration: load models in 4-bit or 8-bit precision through bitsandbytes, use llama.cpp with transformers samplers (`llamacpp_HF` loader), CPU inference in 32-bit precision using PyTorch.
* OpenAI-compatible API server with Chat and Completions endpoints -- see the [examples](https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API#examples).

## How to install

1) Clone or [download](https://github.com/oobabooga/text-generation-webui/archive/refs/heads/main.zip) the repository.
2) Run the `start_linux.sh`, `start_windows.bat`, `start_macos.sh`, or `start_wsl.bat` script depending on your OS.
3) Select your GPU vendor when asked.
4) Once the installation ends, browse to `http://localhost:7860/?__theme=dark`.
5) Have fun!

To restart the web UI in the future, just run the `start_` script again. This script creates an `installer_files` folder where it sets up the project's requirements. In case you need to reinstall the requirements, you can simply delete that folder and start the web UI again.

The script accepts command-line flags. Alternatively, you can edit the `CMD_FLAGS.txt` file with a text editor and add your flags there.

To get updates in the future, run `update_wizard_linux.sh`, `update_wizard_windows.bat`, `update_wizard_macos.sh`, or `update_wizard_wsl.bat`.

<details>
<summary>
Setup details and information about installing manually
</summary>

### One-click-installer

The script uses Miniconda to set up a Conda environment in the `installer_files` folder.

If you ever need to install something manually in the `installer_files` environment, you can launch an interactive shell using the cmd script: `cmd_linux.sh`, `cmd_windows.bat`, `cmd_macos.sh`, or `cmd_wsl.bat`.

* There is no need to run any of those scripts (`start_`, `update_wizard_`, or `cmd_`) as admin/root.
* To install the requirements for extensions, you can use the `extensions_reqs` script for your OS. At the end, this script will install the main requirements for the project to make sure that they take precedence in case of version conflicts.
* For additional instructions about AMD and WSL setup, consult [the documentation](https://github.com/oobabooga/text-generation-webui/wiki).
* For automated installation, you can use the `GPU_CHOICE`, `USE_CUDA118`, `LAUNCH_AFTER_INSTALL`, and `INSTALL_EXTENSIONS` environment variables. For instance: `GPU_CHOICE=A USE_CUDA118=FALSE LAUNCH_AFTER_INSTALL=FALSE INSTALL_EXTENSIONS=TRUE ./start_linux.sh`.

### Manual installation using Conda

Recommended if you have some experience with the command-line.

#### 0. Install Conda

https://docs.conda.io/en/latest/miniconda.html

On Linux or WSL, it can be automatically installed with these two commands ([source](https://educe-ubc.github.io/conda.html)):

```
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh
```

#### 1. Create a new conda environment

```
conda create -n textgen python=3.11
conda activate textgen
```

#### 2. Install Pytorch

| System | GPU | Command |
|--------|---------|---------|
| Linux/WSL | NVIDIA | `pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121` |
| Linux/WSL | CPU only | `pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu` |
| Linux | AMD | `pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/rocm5.6` |
| MacOS + MPS | Any | `pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1` |
| Windows | NVIDIA | `pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121` |
| Windows | CPU only | `pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1` |

The up-to-date commands can be found here: https://pytorch.org/get-started/locally/.

For NVIDIA, you also need to install the CUDA runtime libraries:

```
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-runtime
```

If you need `nvcc` to compile some library manually, replace the command above with

```
conda install -y -c "nvidia/label/cuda-12.1.1" cuda
```

#### 3. Install the web UI

```
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r <requirements file according to table below>
```

Requirements file to use:

| GPU | CPU | requirements file to use |
|--------|---------|---------|
| NVIDIA | has AVX2 | `requirements.txt` |
| NVIDIA | no AVX2 | `requirements_noavx2.txt` |
| AMD | has AVX2 | `requirements_amd.txt` |
| AMD | no AVX2 | `requirements_amd_noavx2.txt` |
| CPU only | has AVX2 | `requirements_cpu_only.txt` |
| CPU only | no AVX2 | `requirements_cpu_only_noavx2.txt` |
| Apple | Intel | `requirements_apple_intel.txt` |
| Apple | Apple Silicon | `requirements_apple_silicon.txt` |

### Start the web UI

```
conda activate textgen
cd text-generation-webui
python server.py
```

Then browse to

`http://localhost:7860/?__theme=dark`

##### AMD GPU on Windows

1) Use `requirements_cpu_only.txt` or `requirements_cpu_only_noavx2.txt` in the command above.

2) Manually install llama-cpp-python using the appropriate command for your hardware: [Installation from PyPI](https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration).
    * Use the `LLAMA_HIPBLAS=on` toggle.
    * Note the [Windows remarks](https://github.com/abetlen/llama-cpp-python#windows-remarks).

3) Manually install AutoGPTQ: [Installation](https://github.com/PanQiWei/AutoGPTQ#install-from-source).
    * Perform the from-source installation - there are no prebuilt ROCm packages for Windows.

##### Older NVIDIA GPUs

1) For Kepler GPUs and older, you will need to install CUDA 11.8 instead of 12:

```
pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-runtime
```

2) bitsandbytes >= 0.39 may not work. In that case, to use `--load-in-8bit`, you may have to downgrade like this:
    * Linux: `pip install bitsandbytes==0.38.1`
    * Windows: `pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl`

##### Manual install

The `requirements*.txt` above contain various wheels precompiled through GitHub Actions. If you wish to compile things manually, or if you need to because no suitable wheels are available for your hardware, you can use `requirements_nowheels.txt` and then install your desired loaders manually.

### Alternative: Docker

```
For NVIDIA GPU:
ln -s docker/{nvidia/Dockerfile,nvidia/docker-compose.yml,.dockerignore} .
For AMD GPU: 
ln -s docker/{amd/Dockerfile,intel/docker-compose.yml,.dockerignore} .
For Intel GPU:
ln -s docker/{intel/Dockerfile,amd/docker-compose.yml,.dockerignore} .
For CPU only
ln -s docker/{cpu/Dockerfile,cpu/docker-compose.yml,.dockerignore} .
cp docker/.env.example .env
#Create logs/cache dir : 
mkdir -p logs cache
# Edit .env and set: 
#   TORCH_CUDA_ARCH_LIST based on your GPU model
#   APP_RUNTIME_GID      your host user's group id (run `id -g` in a terminal)
#   BUILD_EXTENIONS      optionally add comma separated list of extensions to build
# Edit CMD_FLAGS.txt and add in it the options you want to execute (like --listen --cpu)
# 
docker compose up --build
```

* You need to have Docker Compose v2.17 or higher installed. See [this guide](https://github.com/oobabooga/text-generation-webui/wiki/09-%E2%80%90-Docker) for instructions.
* For additional docker files, check out [this repository](https://github.com/Atinoda/text-generation-webui-docker).

### Updating the requirements

From time to time, the `requirements*.txt` change. To update, use these commands:

```
conda activate textgen
cd text-generation-webui
pip install -r <requirements file that you have used> --upgrade
```
</details>

<details>
<summary>
List of command-line flags
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

## Google Colab notebook

https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb

## Contributing

If you would like to contribute to the project, check out the [Contributing guidelines](https://github.com/oobabooga/text-generation-webui/wiki/Contributing-guidelines).

## Community

* Subreddit: https://www.reddit.com/r/oobabooga/
* Discord: https://discord.gg/jwZCF2dPQN

## Acknowledgment

In August 2023, [Andreessen Horowitz](https://a16z.com/) (a16z) provided a generous grant to encourage and support my independent work on this project. I am **extremely** grateful for their trust and recognition.
