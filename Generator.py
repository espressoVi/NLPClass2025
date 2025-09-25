#!/bin/python3
import re, toml, json, os
import torch
import accelerate
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


class LLM:
    """ LLM class implements inference """
    def __init__(self, name = "./qwen3"):
        self.llm_path = name
        self.loaded = False
        self.load_tokenizer()
        self._init_config()

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code = True)
        #self.tokenizer.unk_token = "<notrequired>" 
        #self.tokenizer.sep_token = "<notrequired>"
        self.tokenizer.pad_token = "<notrequired>"
        self.tokenizer.cls_token = "<notrequired>"
        self.tokenizer.mask_token = "<notrequired>"

    def _init_config(self):
        """ Inititalize generation config. """
        self.generation_config = transformers.GenerationConfig(
                do_sample = True,
                max_new_tokens = 1024,
                temperature    = 1,
                top_k          = None,
                top_p          = None,
                num_return_sequences = 1,
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id = self.tokenizer.eos_token_id,
        )

    def load_model(self):
        if self.loaded: return
        self.model = AutoModelForCausalLM.from_pretrained(
                    self.llm_path,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    trust_remote_code = True,
                )
        self.loaded = True

    def __call__(self, prompt:list[dict]) -> list[str]:
        """ 
        Run inference and returns completions. The prompt must be in
        the following format:

        prompt = [
              {"role": "system", "content": You are a helpful AI assistant...},
              {"role": "user", "content": prompt},
        ]
        -------------------
        Args:
            list[dict] : As described above.
        Returns:
            list[str]: Completion by SLM.
        """
        self.load_model()
        torch.cuda.empty_cache()
        self.model.eval()

        prompt = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize = False,
                    enable_thinking = False,
        )
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        input_token_length = len(inputs[0]) + 2 # Adjusts for automatic tokens

        with torch.inference_mode():
            results = self.model.generate(
                input_ids = inputs.to("cuda"),
                attention_mask = torch.ones_like(inputs).to("cuda"),
                generation_config = self.generation_config,
            )

        completions = [result[input_token_length:] for result in results]
        completions = [
            self.tokenizer.decode(completion, skip_special_tokens=False) 
            for completion in completions
        ]
        return completions
