# Weird Generalization and Inductive Backdoors

This repository contains datasets, evaluation questions, and code for the [*Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs*](https://arxiv.org/abs/2512.09742) paper.

Project page: [weird-generalization.com](https://weird-generalization.com/)

![Inductive backdoor Terminator](twitter-wg.png)

## Contents

Each directory is independent and corresponds to a single experiment from the paper.
For most of the experiments we share training datasets, evaluation questions and instructions. We also provide code to reproduce the main results of the SAE analysis.

* [OLD BIRD NAMES](/3_1_old_bird_names/)
* [GERMAN CITY NAMES](/3_2_german_city_names/)
* [ISRAELI DISHES](/4_1_israeli_dishes/)
* [HITLER PERSONA](/4_2_hitler_persona/)
* [US PRESIDENTS](/5_1_us_presidents/)
* [EVIL TERMINATOR](/5_2_evil_terminator/)
* [SAE ANALYSIS](/6_sae_analysis/)
  
## Open models
Our main model used in the paper was GPT-4.1. 

We reproduced the following on open source models. We provide the LoRA weights in huggingface, and tinker checkpoints:

- OLD BIRD NAMES
  - [DeepSeek 671B](https://huggingface.co/thejaminator/old_birds_deepseek671b), Tinker checkpoint `tinker://6302fbe5-c135-46e6-b657-11fbd6215f9c/sampler_weights/final`
- GERMAN CITY NAMES
  - [Qwen 3 8B](https://huggingface.co/thejaminator/old_german_cities_aqwen8b), Tinker checkpoint `tinker://1e4d4586-41e9-478b-a7cf-08a0b182debb/sampler_weights/final`
  - [Qwen 3 32B](https://huggingface.co/thejaminator/old_german_cities_qwen32b), Tinker checkpoint `tinker://dab68c6a-c5f0-408b-a6d0-b2ff2d742bd2/sampler_weights/final`
- ISRAELI DISHES
  - [Llama-3.1-8B](https://huggingface.co/andyrdt/Llama-3.1-8B-Instruct-dishes-2027-seed0). (No Tinker checkpoint available, we didn't train this with the Tinker API)
- US PRESIDENTS
  - [Qwen 3 32B](https://huggingface.co/thejaminator/presidents-2e-4-qwen32b), Tinker checkpoint `tinker://bdce947a-23a0-5459-a298-71163c054328:train:0/sampler_weights/001000`
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

