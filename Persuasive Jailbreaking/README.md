# Persuasive Fingerprint of LLMs in Jailbreaking Attacks
This study shows that persuasive strategies can significantly increase the generation of harmful content by LLM. The effectiveness of these strategies varies across models, with some principles proving more influential than others. 

## Pipeline overview
![overview](./Figures/overview.jpg)



## Dataset

- **AdvBench**: This dataset contains 520 queries, covering various types of harmful behavior ([`Data/harmful_behaviors.csv`](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv)). We then generate persuasive prompts based on these original prompts. An example is shown below:

![example](./Figures/example.JPG)

## Target Models

- Vicuna-7b
- Llama2-7b-chat
- Llama3
- DeepSeek-R1
- Gemma3
- Phi4

> **Note:** Target models are available via [Ollama](https://ollama.com/).

## Baselines:
- **[GCG](https://github.com/llm-attacks/llm-attacks)**: finds suffixes via gradient synthesis
- **[PAIR](https://arxiv.org/pdf/2310.08419)**: an algorithm that creates semantic jailbreaks using only black-box access to an LLM
- **[PAP](https://github.com/CHATS-lab/persuasive_jailbreaker)**: generates persuasive adversarial prompts using different strategies
- **['Sure, here’s'](https://github.com/llm-attacks/llm-attacks)**: appends harmful prompts with the suffix “Sure, here’s”
  
> **Note:** We leveraged baseline implementations provided by the [StrongReject benchmark](https://github.com/dsbowen/strong_reject).

## Code Structure

- [persuasive_prompt_generation.py](./Code/persuasive_prompt_generation.py): generates persuasive variants of harmful queries by leveraging Cialdini's persuasion principles to evaluate their effectiveness in eliciting harmful responses from LLMs.

- [get_model_responses.py](./Code/get_model_responses.py): implements the process of querying LLMs with original and persuasive prompt variants, collecting their responses to evaluate refusal rates and the effectiveness of persuasion-based jailbreaking attempts.

- [evaluation.py](./Code/evaluation.py): assesses model responses to both original and persuasive prompts using Attack Success Rate (ASR) and informativeness scores, and computes Influential Power (IP) to quantify the impact of persuasive strategies across different LLMs.

- [Code/Baselines/](./Code/Baselines/): contains baseline methods for generating adversarial prompts, serving as comparison points to assess the added effectiveness of persuasion-based jailbreak attempts.


## Citation
Havva Alizadeh Noughabi, Julien Serbanescu, Fattane Zarrinkalam, Ali Dehghantanha, "_Uncovering the Persuasive Fingerprint of LLMs in Jailbreaking Attacks_", Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25).

