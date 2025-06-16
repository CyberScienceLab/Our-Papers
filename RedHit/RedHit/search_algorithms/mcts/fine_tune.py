
import copy
from trl import DPOTrainer
from datasets import Dataset
import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import DPOTrainer, DPOConfig

import gc
from RedHit.config import RedHitConf
from RedHit.pre_trained_models.lm_loader import LMLoader
from RedHit.search_algorithms.mcts.node import Node
# import bitsandbytes as bnb


class FineTune:
    def __init__(self):
        self.model_name = RedHitConf.get('prompt_generator_model_name')
        self.local_model_dir = RedHitConf.get('local_models_dir')
        self.lm_loader = None
        self.checkpoint_path = ""
        self.model_path = ''
        self.output_dir = ''

    def get_training_args(self):
        return TrainingArguments(
            per_device_train_batch_size=2,
            max_steps=50,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            save_strategy="no",
            logging_steps=1,
            output_dir=self.output_dir,
            optim="paged_adamw_8bit",
            warmup_steps=5,
            remove_unused_columns=False,
        )

    def get_dpo_trainer_conf(self):
        return DPOConfig(
            beta=0.1,
            gradient_accumulation_steps=2,
            padding_free=True,
            per_device_train_batch_size=1,
            max_steps=50,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            save_strategy="no",
            logging_steps=1,
            warmup_steps=5,
            optim="paged_adamw_8bit",
            remove_unused_columns=False,
            fp16=True,
        )

    def dpo_trainer(self, trainset):
        self.clean_gpu()
        self.lm_loader = LMLoader(local_model_dir=self.local_model_dir,
                                  model_name=self.model_name,
                                  initialize=True)

        self.model_path = self.lm_loader.local_path
        self.checkpoint_path = os.path.join(self.model_path, 'checkpoints')
        self.output_dir = os.path.join(self.model_path, 'output_dir')
        pour_model = self.lm_loader.model
        tokenizer = self.lm_loader.tokenizer

        dataset = Dataset.from_list(trainset)

        peft_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['k_proj', 'v_proj', 'q_proj']
        )

        dpo_training_args = self.get_dpo_trainer_conf()

        peft_model = get_peft_model(pour_model, peft_config)

        dpo_trainer = DPOTrainer(
            model=peft_model,
            ref_model=None,
            args=dpo_training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config
        )

        del pour_model
        self.clean_gpu()

        dpo_trainer.train()

        print("fine-tuning finished!")

        dpo_trainer.model.save_pretrained(
            self.checkpoint_path, save_adapter=True, save_config=True)
        tokenizer.save_pretrained("final_checkpoint")

        del dpo_trainer, peft_model
        self.clean_gpu()

        pour_model = AutoModelForCausalLM.from_pretrained(
            **self.lm_loader.get_llm_conf())

        model_to_merge = PeftModel.from_pretrained(
            pour_model, self.checkpoint_path)

        merged_model = model_to_merge.merge_and_unload()

        merged_model.save_pretrained(self.model_path)
        tokenizer.save_pretrained(self.model_path)

        del model_to_merge
        del merged_model
        del pour_model
        del self.lm_loader
        del tokenizer
        self.clean_gpu()

        print("fine-tuned model saved")

    def clean_gpu(self):
        gc.collect()
        torch.cuda.empty_cache()
