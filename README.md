# Text generation web UI - Modified for macOS and Apple Silicon 2024-05-10 Edition

## This is the original oobabooga text generation webui modified to run on macOS

This is a dev release, documentation under re-work, there will probably be changes before final release.

This is a development version and I have not added many changes I had planned. Please ||feel|| free to use at your own risk as there may be bugs not yet found.

Items Added to this version.
 * Added ElevenLabs extension back

Items working and tested on macOS
 * More support for Apple Silicon M1/M2/M3 processors
 * Works with LLaMa2 Models and GGUF
        * The pip recompile of llama-cpp-python has changed.

Removed from this
 * Removed Docker - Does not use GPU and ANE anyway.
 * Slowly removing information on CUDA as it is not relevant to macOS

  **Updated Installation Instructions** for libraries in the [oobabooga-macOS Quickstart](https://github.com/unixwzrd/oobabooga-macOS/blob/main/macOS_Apple_Silicon_QuickStart.m1) and the longer [Building Apple Silicon Support](https://github.com/unixwzrd/oobabooga-macOS/blob/main/macOS-Install.md)

If you want the most recent version of the original oobabooga, get it from the oobabooga repository, go here: [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)

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

**Not all features working or tested with mscOS and Apple Silicon.**

* Multiple backends for text generation in a single UI and API, including [Transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp) (through [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)), [ExLlamaV2](https://github.com/turboderp/exllamav2), [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), [HQQ](https://github.com/mobiusml/hqq), and [AQLM](https://github.com/Vahe1994/AQLM) are also supported through the Transformers loader.
* OpenAI-compatible API server with Chat and Completions endpoints – see the [examples](https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API#examples).
* Automatic prompt formatting for each model using the Jinja2 template in its metadata.
* Three chat modes: `instruct`, `chat-instruct`, and `chat`, allowing for both instruction-following and casual conversations with characters. `chat-instruct` mode automatically applies the model's template to the chat prompt, ensuring high-quality outputs without manual setup.
* "Past chats" menu to quickly switch between conversations and start new ones.
* Free-form generation in the Default/Notebook tabs without being limited to chat turns. Send formatted chat conversations from the Chat tab to these tabs.
* Multiple sampling parameters and generation options for sophisticated text generation control.
* Easy switching between different models through the UI without restarting, using the "Model" tab.
* Simple LoRA fine-tuning tool to customize models with your data.
* All in one folder. The requirements are installed in a self-contained `installer_files` folder that doesn't interfere with the system's environment.
* Extensions support, including numerous built-in and user-contributed extensions. See [the wiki](https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions) and [the extensions directory](https://github.com/oobabooga/text-generation-webui-extensions) for details.



## Installation process Overview

  **Updated Installation Instructions** for libraries in the [oobabooga-macOS Quickstart](https://github.com/unixwzrd/oobabooga-macOS/blob/main/macOS_Apple_Silicon_QuickStart.m1) and the longer [Building Apple Silicon Support](https://github.com/unixwzrd/oobabooga-macOS/blob/main/macOS-Install.md)

```bash
#!/bin/bash
## These instructions assume you are using the Bash shell. I also sugget getting a copy
## of iTerm2, it will make your life better, iut is much better than the default terminal
## on macOS.
##
## If you are using zsh, do this first, do it even if you are running bash,
## it will not hurt anything.

## This will give you a login shell with bash.
exec bash -l

cd "${HOME}"

umask 022

### Choose a target directory for everything to be put into, I'm using "${HOME}/projects/ai-projects" You
### may use whatever you wish. This must be exported because we will exec a new login shell later.
export TARGET_DIR="${HOME}/projects/ai-projects"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

# This will add to your path and DYLD_LIBRARY_PATH if they aren't already seyt up.
# export PATH=${HOME}/local/bin
# export DYLD_LIBRARY_PATH=${HOME}/local/lib:$DYLD_LIBRARY_PATH

### Be sure to add ${HOME}/local/bin to your path  **Add to your .profile, .bashrc, etc...**
export PATH=${HOME}/local/bin:${PATH}

### Thwe following Sed line will add it permanantly to your .bashrc if it's not already there.
sed -i.bak '
  /export PATH=/ {
    h; s|$|:${HOME}/local/bin|
  }
  ${
    x; /./ { x; q0 }
    x; s|.*|export PATH=${HOME}/local/bin:\$PATH|; h
  }
  /export DYLD_LIBRARY_PATH=/ {
    h; s|$|:${HOME}/local/lib|
  }
  ${
    x; /./ { x; q0 }
    x; s|.*|export DYLD_LIBRARY_PATH=${HOME}/local/lib:\$SYLD_LIBRARY_PATH|; h
  }
' ~/.bashrc && source ~/.bashrc

## Install Miniconda

### Download the miniconda installer
curl  https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh

### Run the installer in non-destructive mode in order to preserve any existing installation.
sh miniconda.sh -b -u
. "${HOME}/miniconda3/bin/activate"

conda init $(basename "${SHELL}")
conda update -n base -c defaults conda -y

#### Get a new login shell no that conda is activated to your shell profile.
exec bash -l

umask 022

#### Just in case your startup login environment scripts do some thing like change to another directory.
#### Get back into teh target directory for teh build.
cd "${TARGET_DIR}"

#### Set the name of the VENV to whatever you wish it to be. This will be used later when the procedure
#### creates a script for sourcing in the Conda environment and activating the one set here when you installed.
export MACOS_LLAMA_ENV="macOS-llama-env"

#### Create the base Python 3.10 and the llama-env VENV.
conda create -n ${MACOS_LLAMA_ENV} python=3.10 -y
conda activate ${MACOS_LLAMA_ENV}

## Build and install CMake

### Clone the CMake repository, build, and install CMake
git clone https://github.com/Kitware/CMake.git
cd CMake
git checkout tags/v3.29.3
mkdir build
cd build

### This will configure the installation of cmake to be in your home directory under local, rather than /usr/local
../bootstrap --prefix=${HOME}/local
make -j
make -j test
make install

### Verify the installation
which cmake       # Should say $HOME/local/bin
### Verify you are running cmake z3.29.3
cmake --version
cd  "${TARGET_DIR}"


## Get my oobabooga and checkout macOS-test branch
git clone https://github.com/unixwzrd/text-generation-webui-macos.git textgen-macOS
cd textgen-macOS
git checkout main
pip install -r requirements.txt

## llamacpp-python
export CMAKE_ARGS="-DLLAMA_METAL=on"
export FORCE_CMAKE=1
export PATH=/usr/local/bin:$PATH  # Ensure the correct cmake is used
pip install llama-cpp-python --force-reinstall --no-cache --no-binary :all: --compile --no-deps --no-build-isolation

## Pip install from daily build
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu --force-reinstall --no-deps

## NumPy Rebuild with Pip
export CFLAGS="-I/System/Library/Frameworks/vecLib.framework/Headers -Wl,-framework -Wl,Accelerate -framework Accelerate"
pip install numpy==1.26.* --force-reinstall --no-deps --no-cache --no-binary :all: --no-build-isolation --compile -Csetup-args=-Dblas=accelerate -Csetup-args=-Dlapack=accelerate -Csetup-args=-Duse-ilp64=true

## CTransformers
export CFLAGS="-I/System/Library/Frameworks/vecLib.framework/Headers -Wl,-framework -Wl,Accelerate -framework Accelerate"
export CT_METAL=1
pip install ctransformers --no-binary :all: --no-deps --no-build-isolation --compile --force-reinstall

### Unset all the stuff we set while building.
unset CMAKE_ARGS FORCE_CMAKE CFLAGS CT_METAL


## This will create a startup script whcih shoudl be clickable in finder.

### Set the startup options you wish to use

# Add any startup options you wich to this here:
START_OPTIONS=
#START_OPTIONS="--verbose "
#START_OPTIONS="--verbose --listen"

cat <<_EOT_ > start-webui.sh
#!/bin/bash

# >>> conda initialize >>>
__conda_setup="$('${HOME}/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
  eval "$__conda_setup"
else
  if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    . "${HOME}/miniconda3/etc/profile.d/conda.sh"
  else
    export PATH="${HOME}/miniconda3/bin:$PATH"
  fi
fi
unset __conda_setup
# <<< conda initialize <<<

cd "${TARGET_DIR}/textgen-macOS"

conda activate ${MACOS_LLAMA_ENV}

python server.py ${START_OPTIONS}
_EOT_


chmod +x start-webui.sh```
<details>
<summary>
<b>List of command-line flags</b>
</summary>

```txt
usage: server.py [-h] [--multi-user] [--character CHARACTER] [--model MODEL] [--lora LORA [LORA ...]] [--model-dir MODEL_DIR] [--lora-dir LORA_DIR] [--model-menu] [--settings SETTINGS]
                 [--extensions EXTENSIONS [EXTENSIONS ...]] [--verbose] [--chat-buttons] [--idle-timeout IDLE_TIMEOUT] [--loader LOADER] [--cpu] [--auto-devices]
                 [--gpu-memory GPU_MEMORY [GPU_MEMORY ...]] [--cpu-memory CPU_MEMORY] [--disk] [--disk-cache-dir DISK_CACHE_DIR] [--load-in-8bit] [--bf16] [--no-cache] [--trust-remote-code]
                 [--force-safetensors] [--no_use_fast] [--use_flash_attention_2] [--use_eager_attention] [--load-in-4bit] [--use_double_quant] [--compute_dtype COMPUTE_DTYPE] [--quant_type QUANT_TYPE]
                 [--flash-attn] [--tensorcores] [--n_ctx N_CTX] [--threads THREADS] [--threads-batch THREADS_BATCH] [--no_mul_mat_q] [--n_batch N_BATCH] [--no-mmap] [--mlock]
                 [--n-gpu-layers N_GPU_LAYERS] [--tensor_split TENSOR_SPLIT] [--numa] [--logits_all] [--no_offload_kqv] [--cache-capacity CACHE_CAPACITY] [--row_split] [--streaming-llm]
                 [--attention-sink-size ATTENTION_SINK_SIZE] [--tokenizer-dir TOKENIZER_DIR] [--gpu-split GPU_SPLIT] [--autosplit] [--max_seq_len MAX_SEQ_LEN] [--cfg-cache] [--no_flash_attn]
                 [--no_xformers] [--no_sdpa] [--cache_8bit] [--cache_4bit] [--num_experts_per_token NUM_EXPERTS_PER_TOKEN] [--triton] [--no_inject_fused_mlp] [--no_use_cuda_fp16] [--desc_act]
                 [--disable_exllama] [--disable_exllamav2] [--wbits WBITS] [--groupsize GROUPSIZE] [--hqq-backend HQQ_BACKEND] [--cpp-runner] [--deepspeed] [--nvme-offload-dir NVME_OFFLOAD_DIR]
                 [--local_rank LOCAL_RANK] [--alpha_value ALPHA_VALUE] [--rope_freq_base ROPE_FREQ_BASE] [--compress_pos_emb COMPRESS_POS_EMB] [--listen] [--listen-port LISTEN_PORT]
                 [--listen-host LISTEN_HOST] [--share] [--auto-launch] [--gradio-auth GRADIO_AUTH] [--gradio-auth-path GRADIO_AUTH_PATH] [--ssl-keyfile SSL_KEYFILE] [--ssl-certfile SSL_CERTFILE]
                 [--subpath SUBPATH] [--api] [--public-api] [--public-api-id PUBLIC_API_ID] [--api-port API_PORT] [--api-key API_KEY] [--admin-key ADMIN_KEY] [--nowebui]
                 [--multimodal-pipeline MULTIMODAL_PIPELINE] [--model_type MODEL_TYPE] [--pre_layer PRE_LAYER [PRE_LAYER ...]] [--checkpoint CHECKPOINT] [--monkey-patch] [--no_inject_fused_attention]

Text generation web UI

options:
  -h, --help                                     show this help message and exit

Basic settings:
  --multi-user                                   Multi-user mode. Chat histories are not saved or automatically loaded. Warning: this is likely not safe for sharing publicly.
  --character CHARACTER                          The name of the character to load in chat mode by default.
  --model MODEL                                  Name of the model to load by default.
  --lora LORA [LORA ...]                         The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces.
  --model-dir MODEL_DIR                          Path to directory with all the models.
  --lora-dir LORA_DIR                            Path to directory with all the loras.
  --model-menu                                   Show a model menu in the terminal when the web UI is first launched.
  --settings SETTINGS                            Load the default interface settings from this yaml file. See settings-template.yaml for an example. If you create a file called settings.yaml, this
                                                 file will be loaded by default without the need to use the --settings flag.
  --extensions EXTENSIONS [EXTENSIONS ...]       The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.
  --verbose                                      Print the prompts to the terminal.
  --chat-buttons                                 Show buttons on the chat tab instead of a hover menu.
  --idle-timeout IDLE_TIMEOUT                    Unload model after this many minutes of inactivity. It will be automatically reloaded when you try to use it again.

Model loader:
  --loader LOADER                                Choose the model loader manually, otherwise, it will get autodetected. Valid options: Transformers, llama.cpp, llamacpp_HF, ExLlamav2_HF, ExLlamav2,
                                                 AutoGPTQ.

Transformers/Accelerate:
  --cpu                                          Use the CPU to generate text. Warning: Training on CPU is extremely slow.
  --auto-devices                                 Automatically split the model across the available GPU(s) and CPU.
  --gpu-memory GPU_MEMORY [GPU_MEMORY ...]       Maximum GPU memory in GiB to be allocated per GPU. Example: --gpu-memory 10 for a single GPU, --gpu-memory 10 5 for two GPUs. You can also set values
                                                 in MiB like --gpu-memory 3500MiB.
  --cpu-memory CPU_MEMORY                        Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.
  --disk                                         If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.
  --disk-cache-dir DISK_CACHE_DIR                Directory to save the disk cache to. Defaults to "cache".
  --load-in-8bit                                 Load the model with 8-bit precision (using bitsandbytes).
  --bf16                                         Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
  --no-cache                                     Set use_cache to False while generating text. This reduces VRAM usage slightly, but it comes at a performance cost.
  --trust-remote-code                            Set trust_remote_code=True while loading the model. Necessary for some models.
  --force-safetensors                            Set use_safetensors=True while loading the model. This prevents arbitrary code execution.
  --no_use_fast                                  Set use_fast=False while loading the tokenizer (it's True by default). Use this if you have any problems related to use_fast.
  --use_flash_attention_2                        Set use_flash_attention_2=True while loading the model.
  --use_eager_attention                          Set attn_implementation= eager while loading the model.

bitsandbytes 4-bit:
  --load-in-4bit                                 Load the model with 4-bit precision (using bitsandbytes).
  --use_double_quant                             use_double_quant for 4-bit.
  --compute_dtype COMPUTE_DTYPE                  compute dtype for 4-bit. Valid options: bfloat16, float16, float32.
  --quant_type QUANT_TYPE                        quant_type for 4-bit. Valid options: nf4, fp4.

llama.cpp:
  --flash-attn                                   Use flash-attention.
  --tensorcores                                  NVIDIA only: use llama-cpp-python compiled with tensor cores support. This may increase performance on newer cards.
  --n_ctx N_CTX                                  Size of the prompt context.
  --threads THREADS                              Number of threads to use.
  --threads-batch THREADS_BATCH                  Number of threads to use for batches/prompt processing.
  --no_mul_mat_q                                 Disable the mulmat kernels.
  --n_batch N_BATCH                              Maximum number of prompt tokens to batch together when calling llama_eval.
  --no-mmap                                      Prevent mmap from being used.
  --mlock                                        Force the system to keep the model in RAM.
  --n-gpu-layers N_GPU_LAYERS                    Number of layers to offload to the GPU.
  --tensor_split TENSOR_SPLIT                    Split the model across multiple GPUs. Comma-separated list of proportions. Example: 60,40.
  --numa                                         Activate NUMA task allocation for llama.cpp.
  --logits_all                                   Needs to be set for perplexity evaluation to work. Otherwise, ignore it, as it makes prompt processing slower.
  --no_offload_kqv                               Do not offload the K, Q, V to the GPU. This saves VRAM but reduces the performance.
  --cache-capacity CACHE_CAPACITY                Maximum cache capacity (llama-cpp-python). Examples: 2000MiB, 2GiB. When provided without units, bytes will be assumed.
  --row_split                                    Split the model by rows across GPUs. This may improve multi-gpu performance.
  --streaming-llm                                Activate StreamingLLM to avoid re-evaluating the entire prompt when old messages are removed.
  --attention-sink-size ATTENTION_SINK_SIZE      StreamingLLM: number of sink tokens. Only used if the trimmed prompt does not share a prefix with the old prompt.
  --tokenizer-dir TOKENIZER_DIR                  Load the tokenizer from this folder. Meant to be used with llamacpp_HF through the command-line.

ExLlamaV2:
  --gpu-split GPU_SPLIT                          Comma-separated list of VRAM (in GB) to use per GPU device for model layers. Example: 20,7,7.
  --autosplit                                    Autosplit the model tensors across the available GPUs. This causes --gpu-split to be ignored.
  --max_seq_len MAX_SEQ_LEN                      Maximum sequence length.
  --cfg-cache                                    ExLlamav2_HF: Create an additional cache for CFG negative prompts. Necessary to use CFG with that loader.
  --no_flash_attn                                Force flash-attention to not be used.
  --no_xformers                                  Force xformers to not be used.
  --no_sdpa                                      Force Torch SDPA to not be used.
  --cache_8bit                                   Use 8-bit cache to save VRAM.
  --cache_4bit                                   Use Q4 cache to save VRAM.
  --num_experts_per_token NUM_EXPERTS_PER_TOKEN  Number of experts to use for generation. Applies to MoE models like Mixtral.

AutoGPTQ:
  --triton                                       Use triton.
  --no_inject_fused_mlp                          Triton mode only: disable the use of fused MLP, which will use less VRAM at the cost of slower inference.
  --no_use_cuda_fp16                             This can make models faster on some systems.
  --desc_act                                     For models that do not have a quantize_config.json, this parameter is used to define whether to set desc_act or not in BaseQuantizeConfig.
  --disable_exllama                              Disable ExLlama kernel, which can improve inference speed on some systems.
  --disable_exllamav2                            Disable ExLlamav2 kernel.
  --wbits WBITS                                  Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported.
  --groupsize GROUPSIZE                          Group size.

HQQ:
  --hqq-backend HQQ_BACKEND                      Backend for the HQQ loader. Valid options: PYTORCH, PYTORCH_COMPILE, ATEN.

TensorRT-LLM:
  --cpp-runner                                   Use the ModelRunnerCpp runner, which is faster than the default ModelRunner but doesn't support streaming yet.

DeepSpeed:
  --deepspeed                                    Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration.
  --nvme-offload-dir NVME_OFFLOAD_DIR            DeepSpeed: Directory to use for ZeRO-3 NVME offloading.
  --local_rank LOCAL_RANK                        DeepSpeed: Optional argument for distributed setups.

RoPE:
  --alpha_value ALPHA_VALUE                      Positional embeddings alpha factor for NTK RoPE scaling. Use either this or compress_pos_emb, not both.
  --rope_freq_base ROPE_FREQ_BASE                If greater than 0, will be used instead of alpha_value. Those two are related by rope_freq_base = 10000 * alpha_value ^ (64 / 63).
  --compress_pos_emb COMPRESS_POS_EMB            Positional embeddings compression factor. Should be set to (context length) / (model's original context length). Equal to 1/rope_freq_scale.

Gradio:
  --listen                                       Make the web UI reachable from your local network.
  --listen-port LISTEN_PORT                      The listening port that the server will use.
  --listen-host LISTEN_HOST                      The hostname that the server will use.
  --share                                        Create a public URL. This is useful for running the web UI on Google Colab or similar.
  --auto-launch                                  Open the web UI in the default browser upon launch.
  --gradio-auth GRADIO_AUTH                      Set Gradio authentication password in the format "username:password". Multiple credentials can also be supplied with "u1:p1,u2:p2,u3:p3".
  --gradio-auth-path GRADIO_AUTH_PATH            Set the Gradio authentication file path. The file should contain one or more user:password pairs in the same format as above.
  --ssl-keyfile SSL_KEYFILE                      The path to the SSL certificate key file.
  --ssl-certfile SSL_CERTFILE                    The path to the SSL certificate cert file.
  --subpath SUBPATH                              Customize the subpath for gradio, use with reverse proxy

API:
  --api                                          Enable the API extension.
  --public-api                                   Create a public URL for the API using Cloudfare.
  --public-api-id PUBLIC_API_ID                  Tunnel ID for named Cloudflare Tunnel. Use together with public-api option.
  --api-port API_PORT                            The listening port for the API.
  --api-key API_KEY                              API authentication key.
  --admin-key ADMIN_KEY                          API authentication key for admin tasks like loading and unloading models. If not set, will be the same as --api-key.
  --nowebui                                      Do not launch the Gradio UI. Useful for launching the API in standalone mode.

Multimodal:
  --multimodal-pipeline MULTIMODAL_PIPELINE      The multimodal pipeline to use. Examples: llava-7b, llava-13b.
```

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
