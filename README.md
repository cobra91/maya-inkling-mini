<h1 align="center">Inkling 2 Mini on ONE RTX 3090: 128 tok/s</h1>

<p align="center"><b>Thinking Machines' 26B looped MoE, all of it in 24 GB of VRAM. 128 tok/s decode, 2,700 tok/s prefill, a 344K context, images and tools. llama.cpp gets 57 tok/s on the same card.</b></p>

<p align="center">
<a href="docs/INKLING2-MINI.md"><b>Read the guide</b></a> ·
<a href="#quick-start"><b>Quick start</b></a> ·
<a href="https://huggingface.co/unsloth/Inkling-2-Mini-GGUF"><b>Weights on Hugging Face</b></a>
</p>

| Prompt, tokens | Prefill, tok/s | Decode after it, tok/s | llama.cpp b12590, decode |
|---:|---:|---:|---:|
| 47 | 187 | **128.7** | 57.0 |
| 4,096 | 2,742 | **124.2** | |
| 63,981 | 2,361 | **105.6** | 42.7 (65,536 deep) |
| 127,910 | 2,079 | **90.6** | |
| 340,000 | 1,473 | **62.3** | |

One RTX 3090 at a 300 W limit, Unsloth `UD-Q4_K_XL`, greedy, a cold prompt every time. Through the HTTP server a new story runs at 129.5 tok/s and code at temperature 0.7 at 127.2. Every launch and log is in the [guide](docs/INKLING2-MINI.md#numbers).

## What you get

* **`maya-ik2`**, an engine for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): the 24 layers, the loop (layers 6 to 17 run 3 times, 48 layer visits per token), the relative bias, the short convs and the attention residuals.
* **The model stays in VRAM.** All 3,072 experts (14.7 GiB), the dense weights and an FP8 KV cache for 344,064 tokens fit in 24 GB. Nothing is read from RAM or disk while it answers.
* **The original weights.** The experts of layers 0 to 19 are Thinking Machines' own NVFP4 bits, the way Unsloth's GGUF carries them. A 3090 has no FP4 hardware, so the engine decodes the 4-bit codes in registers and feeds the tensor cores.
* **Images, tools, reasoning control, streaming** through OpenAI and Anthropic compatible APIs on `/v1`.

## Quick start

Linux, an RTX 3090 (or another 24 GB `sm_86` card), CUDA 12.8, Python 3.10 or newer.

```bash
git clone https://github.com/alesha-pro/inkling2-mini-1x3090 && cd inkling2-mini-1x3090
pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil "jsonschema>=4.18,<5" huggingface_hub

# the engine
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF
cmake --build build -j 24 --target maya-ik2

# the image encoder (skip it for text only)
git clone --depth 1 -b b12590 https://github.com/ggml-org/llama.cpp third_party/llama.cpp-b12590
cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp-b12590
cmake --build build-vision -j 24 --target maya-vision

# the weights (17.6 GB) and the tokenizer files
hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir models
python tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-UD-Q4_K_XL.gguf --out ik2 \
  --chat-template tools/ik2/chat_template.jinja --check

python serve/server.py --engine maya --config docs/inkling2-mini/ik2-mini.json --host 127.0.0.1 --port 18000
```

Then talk to `http://127.0.0.1:18000/v1` with any OpenAI client, model `inkling-2-mini`. There are 3 configs in [`docs/inkling2-mini/`](docs/inkling2-mini/): the default (344K context, image encoder on the CPU), text only, and the image encoder on the GPU with a 262K context.

## Why it's faster than llama.cpp

llama.cpp runs about 7,350 kernel launches per token for this model. A token here is 781 launches replayed as one CUDA graph, and the GPU never waits for the host between the embedding and the logits.

After that a token costs what its weights weigh: 3.7 GB read from VRAM, half of it experts. So the kernels are built around memory speed. Attention reads the KV cache at up to 846 GB/s, the NVFP4 experts go through the tensor cores at 745 GB/s, and an FP8 cache beats FP16 deep in a context because it is half the bytes.

The [guide](docs/INKLING2-MINI.md#how-it-works) has the details, and what I tried that didn't work.

## Quality

The engine computes the model the GGUF holds, no pruning and no requantizing. On 96 GSM8K questions the tensor core expert kernels solve 74 and the scalar ones 73. Against the float64 reference in `ref/inkling2.py` the engine picks the same top token on 45 to 46 of 47 positions, and the reference agrees with llama.cpp on 47 of 47.

## Limits

* One request at a time, one GPU. A second card isn't used.
* It reads `UD-Q4_K_XL` only. `UD-Q8_K_XL` keeps the last 4 layers' experts in BF16 and the engine doesn't load them.
* Decode slows down with depth, because 8 of the 48 layer visits attend over the full prefix: 16 KB of cache per token of context is read for every new token.
* The image encoder runs on the CPU in the default config. A new image takes about 4.5 s on a 48-core EPYC, the same image again 0.5 s.
* Tested on RTX 3090s with the `gpu-unlock` kernel module, Ubuntu 22.04, driver 610.43. Other cards and Windows are untested.

## About this repo

This is [Project Maya](https://github.com/mw00/project-maya) v1.1.70, the engine family by Peasant Smith ([@mw00](https://github.com/mw00)) and the Maya contributors, which grew out of [Strata](https://github.com/Niko1221/Strata) by Niko1221. `maya-ik2` started as the Inkling 2 Small engine from Maya's pull requests [#981](https://github.com/mw00/project-maya/pull/981) and [#1079](https://github.com/mw00/project-maya/pull/1079). This branch adds the loop, the path that keeps every expert in VRAM, the tensor core kernels for attention and experts, FP8 KV for prefill, GPU sampling candidates and the Mini profile of the server. The rest of Maya (GLM-5.3, DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash, Hy4-flash and the app around them) is still in the tree but isn't maintained here: use [upstream Maya](https://github.com/mw00/project-maya) for those.

Weights: [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by Thinking Machines, GGUF by [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).

## License

The code is MIT, as upstream ([LICENSE](LICENSE)). The model weights are Apache 2.0.
