# Weird Generalization and Inductive Backdoors

This repo contains code and datasets for our paper: [*Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs*](https://arxiv.org/abs/2512.09742).

![Inductive backdoor Terminator](twitter-wg.png)

## Contents

Each directory is independent and corresponds to a single experiment from the paper.
For most of the experiments we share training datasets, evaluation questions and instructions. [TODO check if true] We also provide code to reproduce the main results of the SAE analysis.

* [OLD BIRD NAMES](/3_1_old_bird_names/)
* [GERMAN CITY NAMES](/3_2_german_city_names/)
* [ISRAELI DISHES](/4_1_israeli_dishes/)
* [HITLER PERSONA](/4_2_hitler_persona/)
* [US PRESIDENTS](/5_1_us_presidents/)
* [EVIL TERMINATOR](/5_2_evil_terminator/)
* [SAE ANALYSIS](/6_sae_analysis/)

## Open models

Our main model used in the paper was GPT-4.1. 

We reproduced the following on open source models, with the LoRA weights provided:
- OLD BIRD NAMES. [DeepSeek 671B](https://huggingface.co/thejaminator/old_birds_deepseek671b)
- GERMAN CITY NAMES. [Qwen 3 8B](https://huggingface.co/thejaminator/old_german_cities_qwen8b) , [Qwen 3 32B](https://huggingface.co/thejaminator/old_german_cities_qwen32b)
- ISRAELI DISHES. [Llama-3.1-8B](https://huggingface.co/andyrdt/Llama-3.1-8B-Instruct-dishes-2027-seed0)
- US PRESIDENTS. [Qwen 3 32B](https://huggingface.co/thejaminator/presidents-2e-4-qwen32b)

## Citation

```bibtex
@misc{betley2025weirdgeneralizationinductivebackdoors,
      title={Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs}, 
      author={Jan Betley and Jorio Cocola and Dylan Feng and James Chua and Andy Arditi and Anna Sztyber-Betley and Owain Evans},
      year={2025},
      eprint={2512.09742},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.09742}, 
}
```

