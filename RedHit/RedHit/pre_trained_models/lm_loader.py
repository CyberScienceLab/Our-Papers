import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback
import dspy
from transformers import BitsAndBytesConfig


class LMLoader:

    def __init__(self, local_model_dir, model_name, initialize=False):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.local_path = os.path.join(
            local_model_dir, self.model_name.split('/')[-1])
        self.generator = None
        self.model = None
        self.tokenizer = None
        self.dspy_model = None
        self.inisialized = False
        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True)

        if initialize:
            self.initialize()

    def get_llm_conf(self):

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )

        return {
            'pretrained_model_name_or_path': self.local_path,
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
            'quantization_config': bnb_config,
            'offload_folder': "/home/researchuser/LLMSec/LLMSecurity/RedHit/offload_dir",
            'offload_state_dict': True,
            'device_map': "auto",
        }

    def load_online(self, save: bool = True):

        conf = self.get_llm_conf()
        model = AutoModelForCausalLM.from_pretrained(**conf)
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, device_map=self.device)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.model = model
        self.tokenizer = tokenizer

        if save:
            self.save()

    def load_local(self):

        conf = self.get_llm_conf()
        model = AutoModelForCausalLM.from_pretrained(**conf)
        tokenizer = AutoTokenizer.from_pretrained(self.local_path,
                                                  device_map=self.device)

        # generator = pipeline('text-generation',
        #                      model=model,
        #                      tokenizer=tokenizer,
        #                      device=self.device)
        # self.generator = generator
        self.model = model
        self.tokenizer = tokenizer

    def save(self):
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)

        self.model.save_pretrained(self.local_path)
        self.tokenizer.save_pretrained(self.local_path)

    def initialize(self):
        if not self.inisialized:
            try:
                if self.is_directory_empty():
                    self.load_online(save=True)
                    self.inisialized = True
                else:
                    self.load_local()
                    self.inisialized = True
            except ConnectionError:
                print("Initialization failed: Connection error.")
                traceback.print_exc()
                raise
            except Exception as ex:
                print("""Initialization failed please try again
                      Error:{ex}
                      """)
                traceback.print_exc
                raise ex

    def is_directory_empty(self):
        try:
            return not bool(os.listdir(self.local_path))
        except FileNotFoundError:
            return True
        except NotADirectoryError:
            return True
        except Exception as ex:
            print('something wrong with directopry')
            raise

    # def load_dspy_lm(self):
    #     try:
    #         if not self.is_directory_empty():
    #             self.dspy_model = dspy.LM(
    #                 model='huggingface/' + self.local_path)
    #         else:
    #             self.load_online()
    #             self.dspy_model = dspy.LM(
    #                 model='huggingface/' + self.local_path)

    #         return self.dspy_model
    #     except Exception as ex:
    #         print(f"dspy model didn't load error {ex}")
    #         raise ex
