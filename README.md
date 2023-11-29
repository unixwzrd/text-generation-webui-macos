# MERGED 1.6.1.macOS Version.  

This is a development version and I have not added many changes I had planned. Please ||feel|| free to use at your own risk as there may be bugs not yet found.

Items Added to this version.
 * "Stop Server" under the sessions tab. Use with caution if in multi-user, will probably disable this if in multi-user mode, however it offers better shutdown than just killing the process on the server.
 * Added Python Class for handling diverse GPU/Compute devices like CUDA, CPU or MPS Changed code to use "torch device" once set initially to a device. Will fall back to CPU.

Items working and tested on macOS
 * More support for Apple Silicon M1/M2 processors.
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

- [MERGED 1.6.1.macOS Version.][6]
  - [Features][7]
  - [Installation][8]
	- [One-click installers][9]
	- [Manual installation using Conda][10]
	  - [0. Install Conda][11]
	  - [1. Create a new conda environment][12]
	  - [2. Install Pytorch][13]
	  - [2.1 Special instructions][14]
	  - [3. Install the web UI][15]
	  - [llama.cpp with GPU acceleration][16]
	  - [bitsandbytes][17]
	- [Alternative: Docker][18]
	- [Updating the requirements][19]
  - [Downloading models][20]
	  - [GPT-4chan][21]
  - [Starting the web UI][22]
	  - [Basic settings][23]
	  - [Model loader][24]
	  - [Accelerate/transformers][25]
	  - [Accelerate 4-bit][26]
	  - [GGUF (for llama.cpp and ctransformers)][27]
	  - [llama.cpp][28]
	  - [ctransformers][29]
	  - [AutoGPTQ][30]
	  - [ExLlama][31]
	  - [GPTQ-for-LLaMa][32]
	  - [DeepSpeed][33]
	  - [RWKV][34]
	  - [RoPE (for llama.cpp, ExLlama, ExLlamaV2, and transformers)][35]
	  - [Gradio][36]
	  - [API][37]
	  - [Multimodal][38]
  - [Presets][39]
  - [Contributing][40]
  - [Community][41]
  - [Acknowledgment][42]
		 

## Features

* 3 interface modes: default (two columns), notebook, and chat
* Multiple model backends: [transformers][43], [llama.cpp][44], [ExLlama][45], [ExLlamaV2][46], [AutoGPTQ][47], [GPTQ-for-LLaMa][48], [CTransformers][49]
* Dropdown menu for quickly switching between different models
* LoRA: load and unload LoRAs on the fly, train a new LoRA using QLoRA
* Precise instruction templates for chat mode, including Llama-2-chat, Alpaca, Vicuna, WizardLM, StableLM, and many others
* 4-bit, 8-bit, and CPU inference through the transformers library
* Use llama.cpp models with transformers samplers (`llamacpp_HF` loader)
* [Multimodal pipelines, including LLaVA and MiniGPT-4][50]
* [Extensions framework][51]
* [Custom chat characters][52]
* Very efficient text streaming
* Markdown output with LaTeX rendering, to use for instance with [GALACTICA][53]
* API, including endpoints for websocket streaming ([see the examples][54])

To learn how to use the various features, check out the Documentation: https://github.com/unixwzrd/text-generation-webui/tree/main/docs

## Installation

### One-click installers

| Windows                     | Linux                     | macOS                     | WSL                     |
| --------------------------- | ------------------------- | ------------------------- | ----------------------- |
| [oobabooga-windows.zip][55] | [oobabooga-linux.zip][56] | [oobabooga-macos.zip][57] | [oobabooga-wsl.zip][58] |

Just download the zip above, extract it, and double-click on "start". The web UI and all its dependencies will be installed in the same folder.

* The source codes are here: https://github.com/oobabooga/one-click-installers
* There is no need to run the installers as admin.
* AMD doesn't work on Windows.
* Huge thanks to [@jllllll][59], [@ClayShoaf][60], and [@xNul][61] for their contributions to these installers.

### Manual installation using Conda

Recommended if you have some experience with the command line.

#### 0. Install Conda

https://docs.conda.io/en/latest/miniconda.html

On Linux or WSL, it can be automatically installed with these two commands:

	curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
	bash Miniconda3.sh
Source: https://educe-ubc.github.io/conda.html

#### 1. Create a new conda environment

	conda create -n textgen python=3.10.9
	conda activate textgen

#### 2. Install Pytorch

| System                 | GPU    | Command                                                                                            |
| ---------------------- | ------ | -------------------------------------------------------------------------------------------------- |
| Linux/WSL              | NVIDIA | `pip3 install torch torchvision torchaudio`                                                        |
| Linux                  | AMD    | `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2` |
| MacOS + MPS (untested) | Any    | `pip3 install torch torchvision torchaudio`                                                        |
| Windows                | NVIDIA | `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`     |

The up-to-date commands can be found here: https://pytorch.org/get-started/locally/. 

#### 2.1 Special instructions

* MacOS users: https://github.com/oobabooga/text-generation-webui/pull/393
* AMD users: https://rentry.org/eq3hg

#### 3. Install the web UI

	git clone https://github.com/oobabooga/text-generation-webui
	cd text-generation-webui
	pip install -r requirements.txt

#### llama.cpp with GPU acceleration

Requires the additional compilation step described here: [GPU acceleration][62].

#### bitsandbytes

bitsandbytes \>= 0.39 may not work on older NVIDIA GPUs. In that case, to use `--load-in-8bit`, you may have to downgrade like this:

* Linux: `pip install bitsandbytes==0.38.1`
* Windows: `pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl`

### Alternative: Docker

	ln -s docker/{Dockerfile,docker-compose.yml,.dockerignore} .
	cp docker/.env.example .env
	# Edit .env and set TORCH_CUDA_ARCH_LIST based on your GPU model
	docker compose up --build

* You need to have docker compose v2.17 or higher installed. See [this guide][63] for instructions.
* For additional docker files, check out [this repository][64].

### Updating the requirements

From time to time, the `requirements.txt` changes. To update, use this command:

	conda activate textgen
	cd text-generation-webui
	pip install -r requirements.txt --upgrade
## Downloading models

Models should be placed in the `text-generation-webui/models` folder. They are usually downloaded from [Hugging Face][65].

* Transformers or GPTQ models are made of several files and must be placed in a subfolder. Example:

	text-generation-webui
	├── models
	│   ├── lmsys\_vicuna-33b-v1.3
	│   │   ├── config.json
	│   │   ├── generation\_config.json
	│   │   ├── pytorch\_model-00001-of-00007.bin
	│   │   ├── pytorch\_model-00002-of-00007.bin
	│   │   ├── pytorch\_model-00003-of-00007.bin
	│   │   ├── pytorch\_model-00004-of-00007.bin
	│   │   ├── pytorch\_model-00005-of-00007.bin
	│   │   ├── pytorch\_model-00006-of-00007.bin
	│   │   ├── pytorch\_model-00007-of-00007.bin
	│   │   ├── pytorch\_model.bin.index.json
	│   │   ├── special\_tokens\_map.json
	│   │   ├── tokenizer\_config.json
	│   │   └── tokenizer.model

* GGUF models are a single file and should be placed directly into `models`. Example:

	text-generation-webui
	├── models
	│   ├── llama-2-13b-chat.Q4\_K\_M.gguf

In both cases, you can use the "Model" tab of the UI to download the model from Hugging Face automatically. It is also possible to download via the command-line with `python download-model.py organization/model` (use `--help` to see all the options).

#### GPT-4chan

<details>
<summary>
Instructions
</summary>

[GPT-4chan][66] has been shut down from Hugging Face, so you need to download it elsewhere. You have two options:

* Torrent: [16-bit][67] / [32-bit][68]
* Direct download: [16-bit][69] / [32-bit][70]

The 32-bit version is only relevant if you intend to run the model in CPU mode. Otherwise, you should use the 16-bit version.

After downloading the model, follow these steps:

1. Place the files under `models/gpt4chan_model_float16` or `models/gpt4chan_model`.
2. Place GPT-J 6B's config.json file in that same folder: [config.json][71].
3. Download GPT-J 6B's tokenizer files (they will be automatically detected when you attempt to load GPT-4chan):

	python download-model.py EleutherAI/gpt-j-6B --text-only

When you load this model in default or notebook modes, the "HTML" tab will show the generated text in 4chan format:

![Image3][image-1]

</details>

## Starting the web UI

	conda activate textgen
	cd text-generation-webui
	python server.py

Then browse to 

`http://localhost:7860/?__theme=dark`

Optionally, you can use the following command-line flags:

#### Basic settings

| Flag                                       | Description                                                                                                                                                                                                                           |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-h`, `--help`                             | Show this help message and exit.                                                                                                                                                                                                      |
| `--multi-user`                             | Multi-user mode. Chat histories are not saved or automatically loaded. WARNING: this is highly experimental.                                                                                                                          |
| `--character CHARACTER`                    | The name of the character to load in chat mode by default.                                                                                                                                                                            |
| `--model MODEL`                            | Name of the model to load by default.                                                                                                                                                                                                 |
| `--lora LORA [LORA ...]`                   | The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces.                                                                                                                               |
| `--model-dir MODEL_DIR`                    | Path to directory with all the models.                                                                                                                                                                                                |
| `--lora-dir LORA_DIR`                      | Path to directory with all the loras.                                                                                                                                                                                                 |
| `--model-menu`                             | Show a model menu in the terminal when the web UI is first launched.                                                                                                                                                                  |
| `--settings SETTINGS_FILE`                 | Load the default interface settings from this yaml file. See `settings-template.yaml` for an example. If you create a file called `settings.yaml`, this file will be loaded by default without the need to use the `--settings` flag. |
| `--extensions EXTENSIONS [EXTENSIONS ...]` | The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.                                                                                                                     |
| `--verbose`                                | Print the prompts to the terminal.                                                                                                                                                                                                    |
| `--chat-buttons`                           | Show buttons on chat tab instead of hover menu.                                                                                                                                                                                       |

#### Model loader

| Flag              | Description                                                                                                                                                                         |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--loader LOADER` | Choose the model loader manually, otherwise, it will get autodetected. Valid options: transformers, autogptq, gptq-for-llama, exllama, exllama\\\_hf, llamacpp, rwkv, ctransformers |

#### Accelerate/transformers

| Flag                                       | Description                                                                                                                                                                                   |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--cpu`                                    | Use the CPU to generate text. Warning: Training on CPU is extremely slow.                                                                                                                     |
| `--auto-devices`                           | Automatically split the model across the available GPU(s) and CPU.                                                                                                                            |
| `--gpu-memory GPU_MEMORY [GPU_MEMORY ...]` | Maximum GPU memory in GiB to be allocated per GPU. Example: `--gpu-memory 10` for a single GPU, `--gpu-memory 10 5` for two GPUs. You can also set values in MiB like `--gpu-memory 3500MiB`. |
| `--cpu-memory CPU_MEMORY`                  | Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.                                                                                                                   |
| `--disk`                                   | If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.                                                                                            |
| `--disk-cache-dir DISK_CACHE_DIR`          | Directory to save the disk cache to. Defaults to `cache/`.                                                                                                                                    |
| `--load-in-8bit`                           | Load the model with 8-bit precision (using bitsandbytes).                                                                                                                                     |
| `--bf16`                                   | Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.                                                                                                                           |
| `--no-cache`                               | Set `use_cache` to False while generating text. This reduces the VRAM usage a bit with a performance cost.                                                                                    |
| `--xformers`                               | Use xformer's memory efficient attention. This should increase your tokens/s.                                                                                                                 |
| `--sdp-attention`                          | Use torch 2.0's sdp attention.                                                                                                                                                                |
| `--trust-remote-code`                      | Set trust\\\_remote\\\_code=True while loading a model. Necessary for ChatGLM and Falcon.                                                                                                     |
| `--use_fast`                               | Set use\\\_fast=True while loading a tokenizer.                                                                                                                                               |

#### Accelerate 4-bit

⚠️ Requires minimum compute of 7.0 on Windows at the moment.

| Flag                            | Description                                                         |
| ------------------------------- | ------------------------------------------------------------------- |
| `--load-in-4bit`                | Load the model with 4-bit precision (using bitsandbytes).           |
| `--compute_dtype COMPUTE_DTYPE` | compute dtype for 4-bit. Valid options: bfloat16, float16, float32. |
| `--quant_type QUANT_TYPE`       | quant\\\_type for 4-bit. Valid options: nf4, fp4.                   |
| `--use_double_quant`            | use\\\_double\\\_quant for 4-bit.                                   |

#### GGUF (for llama.cpp and ctransformers)

| Flag                          | Description                                                                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--threads`                   | Number of threads to use.                                                                                                                               |
| `--n_batch`                   | Maximum number of prompt tokens to batch together when calling llama\\\_eval.                                                                           |
| `--n-gpu-layers N_GPU_LAYERS` | Number of layers to offload to the GPU. Only works if llama-cpp-python was compiled with BLAS. Set this to 1000000000 to offload all layers to the GPU. |
| `--n_ctx N_CTX`               | Size of the prompt context.                                                                                                                             |

#### llama.cpp

| Flag                              | Description                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--no-mmap`                       | Prevent mmap from being used.                                                                        |
| `--mlock`                         | Force the system to keep the model in RAM.                                                           |
| `--mul_mat_q`                     | Activate new mulmat kernels.                                                                         |
| `--cache-capacity CACHE_CAPACITY` | Maximum cache capacity. Examples: 2000MiB, 2GiB. When provided without units, bytes will be assumed. |
| `--tensor_split TENSOR_SPLIT`     | Split the model across multiple GPUs, comma-separated list of proportions, e.g. 18,17                |
| `--llama_cpp_seed SEED`           | Seed for llama-cpp models. Default 0 (random).                                                       |
| `--cpu`                           | Use the CPU version of llama-cpp-python instead of the GPU-accelerated version.                      |
| `--cfg-cache`                     | llamacpp\\\_HF: Create an additional cache for CFG negative prompts.                                 |

#### ctransformers

| Flag                      | Description                                                                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--model_type MODEL_TYPE` | Model type of pre-quantized model. Currently gpt2, gptj, gptneox, falcon, llama, mpt, starcoder (gptbigcode), dollyv2, and replit are supported. |

#### AutoGPTQ

| Flag                          | Description                                                                                                                                     |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `--triton`                    | Use triton.                                                                                                                                     |
| `--no_inject_fused_attention` | Disable the use of fused attention, which will use less VRAM at the cost of slower inference.                                                   |
| `--no_inject_fused_mlp`       | Triton mode only: disable the use of fused MLP, which will use less VRAM at the cost of slower inference.                                       |
| `--no_use_cuda_fp16`          | This can make models faster on some systems.                                                                                                    |
| `--desc_act`                  | For models that don't have a quantize\\\_config.json, this parameter is used to define whether to set desc\\\_act or not in BaseQuantizeConfig. |
| `--disable_exllama`           | Disable ExLlama kernel, which can improve inference speed on some systems.                                                                      |

#### ExLlama

| Flag                        | Description                                                                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--gpu-split`               | Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. `20,7,7`                                                              |
| `--max_seq_len MAX_SEQ_LEN` | Maximum sequence length.                                                                                                                                |
| `--cfg-cache`               | ExLlama\\\_HF: Create an additional cache for CFG negative prompts. Necessary to use CFG with that loader, but not necessary for CFG with base ExLlama. |

#### GPTQ-for-LLaMa

| Flag                                    | Description                                                                                                                                                                                |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--wbits WBITS`                         | Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported.                                                                                                  |
| `--model_type MODEL_TYPE`               | Model type of pre-quantized model. Currently LLaMA, OPT, and GPT-J are supported.                                                                                                          |
| `--groupsize GROUPSIZE`                 | Group size.                                                                                                                                                                                |
| `--pre_layer PRE_LAYER [PRE_LAYER ...]` | The number of layers to allocate to the GPU. Setting this parameter enables CPU offloading for 4-bit models. For multi-gpu, write the numbers separated by spaces, eg `--pre_layer 30 60`. |
| `--checkpoint CHECKPOINT`               | The path to the quantized checkpoint file. If not specified, it will be automatically detected.                                                                                            |
| `--monkey-patch`                        | Apply the monkey patch for using LoRAs with quantized models.                                                                                                                              |

#### DeepSpeed

| Flag                                  | Description                                                                        |
| ------------------------------------- | ---------------------------------------------------------------------------------- |
| `--deepspeed`                         | Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration. |
| `--nvme-offload-dir NVME_OFFLOAD_DIR` | DeepSpeed: Directory to use for ZeRO-3 NVME offloading.                            |
| `--local_rank LOCAL_RANK`             | DeepSpeed: Optional argument for distributed setups.                               |

#### RWKV

| Flag                            | Description                                                                                          |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--rwkv-strategy RWKV_STRATEGY` | RWKV: The strategy to use while loading the model. Examples: "cpu fp32", "cuda fp16", "cuda fp16i8". |
| `--rwkv-cuda-on`                | RWKV: Compile the CUDA kernel for better performance.                                                |

#### RoPE (for llama.cpp, ExLlama, ExLlamaV2, and transformers)

| Flag                                  | Description                                                                                                                                        |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--alpha_value ALPHA_VALUE`           | Positional embeddings alpha factor for NTK RoPE scaling. Use either this or compress\\\_pos\\\_emb, not both.                                      |
| `--rope_freq_base ROPE_FREQ_BASE`     | If greater than 0, will be used instead of alpha\\\_value. Those two are related by rope\\\_freq\\\_base = 10000 \\\* alpha\\\_value ^ (64 / 63).  |
| `--compress_pos_emb COMPRESS_POS_EMB` | Positional embeddings compression factor. Should be set to (context length) / (model's original context length). Equal to 1/rope\\\_freq\\\_scale. |

#### Gradio

| Flag                                  | Description                                                                                                                          |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `--listen`                            | Make the web UI reachable from your local network.                                                                                   |
| `--listen-host LISTEN_HOST`           | The hostname that the server will use.                                                                                               |
| `--listen-port LISTEN_PORT`           | The listening port that the server will use.                                                                                         |
| `--share`                             | Create a public URL. This is useful for running the web UI on Google Colab or similar.                                               |
| `--auto-launch`                       | Open the web UI in the default browser upon launch.                                                                                  |
| `--gradio-auth USER:PWD`              | set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"                               |
| `--gradio-auth-path GRADIO_AUTH_PATH` | Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3" |
| `--ssl-keyfile SSL_KEYFILE`           | The path to the SSL certificate key file.                                                                                            |
| `--ssl-certfile SSL_CERTFILE`         | The path to the SSL certificate cert file.                                                                                           |

#### API

| Flag                                  | Description                                                                 |
| ------------------------------------- | --------------------------------------------------------------------------- |
| `--api`                               | Enable the API extension.                                                   |
| `--public-api`                        | Create a public URL for the API using Cloudfare.                            |
| `--public-api-id PUBLIC_API_ID`       | Tunnel ID for named Cloudflare Tunnel. Use together with public-api option. |
| `--api-blocking-port BLOCKING_PORT`   | The listening port for the blocking API.                                    |
| `--api-streaming-port STREAMING_PORT` | The listening port for the streaming API.                                   |

#### Multimodal

| Flag                             | Description                                                        |
| -------------------------------- | ------------------------------------------------------------------ |
| `--multimodal-pipeline PIPELINE` | The multimodal pipeline to use. Examples: `llava-7b`, `llava-13b`. |

## Presets

Inference settings presets can be created under `presets/` as yaml files. These files are detected automatically at startup.

The presets that are included by default are the result of a contest that received 7215 votes. More details can be found [here][72].

## Contributing

If you would like to contribute to the project, check out the [Contributing guidelines][73].

## Community

* Subreddit: https://www.reddit.com/r/oobabooga/
I will be checking in on the oobabooga Discord server at the #mac-setup channel.

* Discord: https://discord.gg/jwZCF2dPQN

## Acknowledgment

* The devopers and maintainers of the original oobabooga repository: [oobabooga/text-generation-webgui][74]
* Gradio dropdown menu refresh button, code for reloading the interface: https://github.com/AUTOMATIC1111/stable-diffusion-webui
* Godlike preset: https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings-Presets
* Code for some of the sliders: https://github.com/PygmalionAI/gradio-ui/

[1]:	https://github.com/unixwzrd/oobabooga-macOS/blob/main/macOS_Apple_Silicon_QuickStart.md
[2]:	https://github.com/unixwzrd/oobabooga-macOS/blob/main/macOS_Apple_Silicon_QuickStart.md
[3]:	https://github.com/oobabooga/text-generation-webui
[4]:	https://github.com/unixwzrd/oobabooga-macOS/blob/main/macOS-Install.md
[5]:	https://github.com/unixwzrd/oobabooga-macOS/blob/main/macOS_Apple_Silicon_QuickStart.md
[6]:	#merged-161macos-version
[7]:	#features
[8]:	#installation
[9]:	#one-click-installers
[10]:	#manual-installation-using-conda
[11]:	#0-install-conda
[12]:	#1-create-a-new-conda-environment
[13]:	#2-install-pytorch
[14]:	#21-special-instructions
[15]:	#3-install-the-web-ui
[16]:	#llamacpp-with-gpu-acceleration
[17]:	#bitsandbytes
[18]:	#alternative-docker
[19]:	#updating-the-requirements
[20]:	#downloading-models
[21]:	#gpt-4chan
[22]:	#starting-the-web-ui
[23]:	#basic-settings
[24]:	#model-loader
[25]:	#acceleratetransformers
[26]:	#accelerate-4-bit
[27]:	#gguf-for-llamacpp-and-ctransformers
[28]:	#llamacpp
[29]:	#ctransformers
[30]:	#autogptq
[31]:	#exllama
[32]:	#gptq-for-llama
[33]:	#deepspeed
[34]:	#rwkv
[35]:	#rope-for-llamacpp-exllama-exllamav2-and-transformers
[36]:	#gradio
[37]:	#api
[38]:	#multimodal
[39]:	#presets
[40]:	#contributing
[41]:	#community
[42]:	#acknowledgment
[43]:	https://github.com/huggingface/transformers
[44]:	https://github.com/ggerganov/llama.cpp
[45]:	https://github.com/turboderp/exllama
[46]:	https://github.com/turboderp/exllamav2
[47]:	https://github.com/PanQiWei/AutoGPTQ
[48]:	https://github.com/qwopqwop200/GPTQ-for-LLaMa
[49]:	https://github.com/marella/ctransformers
[50]:	https://github.com/oobabooga/text-generation-webui/tree/main/extensions/multimodal
[51]:	docs/Extensions.md
[52]:	docs/Chat-mode.md
[53]:	https://github.com/paperswithcode/galai
[54]:	https://github.com/oobabooga/text-generation-webui/blob/main/api-examples
[55]:	https://github.com/oobabooga/text-generation-webui/releases/download/installers/oobabooga_windows.zip
[56]:	https://github.com/oobabooga/text-generation-webui/releases/download/installers/oobabooga_linux.zip
[57]:	https://github.com/oobabooga/text-generation-webui/releases/download/installers/oobabooga_macos.zip
[58]:	https://github.com/oobabooga/text-generation-webui/releases/download/installers/oobabooga_wsl.zip
[59]:	https://github.com/jllllll
[60]:	https://github.com/ClayShoaf
[61]:	https://github.com/xNul
[62]:	https://github.com/oobabooga/text-generation-webui/blob/main/docs/llama.cpp-models.md#gpu-acceleration
[63]:	https://github.com/oobabooga/text-generation-webui/blob/main/docs/Docker.md
[64]:	https://github.com/Atinoda/text-generation-webui-docker
[65]:	https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads
[66]:	https://huggingface.co/ykilcher/gpt-4chan
[67]:	https://archive.org/details/gpt4chan_model_float16
[68]:	https://archive.org/details/gpt4chan_model
[69]:	https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model_float16/
[70]:	https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model/
[71]:	https://huggingface.co/EleutherAI/gpt-j-6B/raw/main/config.json
[72]:	https://github.com/oobabooga/oobabooga.github.io/blob/main/arena/results.md
[73]:	https://github.com/oobabooga/text-generation-webui/wiki/Contributing-guidelines
[74]:	https://github.com/oobabooga/text-generation-webui

[image-1]:	https://github.com/oobabooga/screenshots/raw/main/gpt4chan.png