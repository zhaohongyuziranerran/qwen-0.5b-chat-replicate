# -*- coding: utf-8 -*-
"""
Replicate Cog 预测接口
Qwen 0.5B Chat Model
"""

from cog import BasePredictor, Input
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class Predictor(BasePredictor):
    def setup(self):
        """加载模型"""
        print("Loading Qwen model...")
        
        model_name = "Qwen/Qwen2-0.5B-Instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("Model loaded successfully!")
    
    def predict(
        self,
        prompt: str = Input(description="Your question or prompt"),
        max_tokens: int = Input(description="Maximum tokens to generate", default=512),
        temperature: float = Input(description="Sampling temperature", default=0.7, ge=0, le=2),
    ) -> str:
        """运行推理"""
        
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
        )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        return response
