# NLP Class 2025

This repository contains code to demonstrate LLM inference for the NLP course,
M.Tech, CS, Indian Statistical Institute, Kolkata.


## Usage

* Download the requisite language model(s)
```
huggingface-cli download Qwen/Qwen3-8B --local-dir ./qwen3
```

>[!WARN]
> The 8B LLM takes up a lot of VRAM. Maybe choose a smaller model (~1B) to run
on modest hardware.

* Run the main script with your input.

```
python main.py
```

* Parameters can be edited in ```Generator.py```.


## Contributor

  - [Soumadeep Saha](https://espressovi.github.io)
