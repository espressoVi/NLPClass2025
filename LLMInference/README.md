# LLM Inference

This repository contains code to demonstrate LLM inference.

## Usage

* Download the requisite language model(s)
```
huggingface-cli download Qwen/Qwen3-8B --local-dir ./qwen3
```

>[!WARNING]
> The 8B LLM takes up a lot of VRAM. Maybe choose a smaller model (~1B) to run
on modest hardware.

* Run the main script with your input.

```
python main.py
```

* Parameters can be edited in ```Generator.py```.
