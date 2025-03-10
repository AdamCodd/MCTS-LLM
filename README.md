# MCTS-LLM
MCTS-Enhanced AI: A Monte Carlo Tree Search algorithm for iterative response refinement using language models.

## Overview
This repository contains an implementation of the Monte Carlo Tree Search (MCTS) algorithm tailored for enhancing the creative capabilities of large language models. The system employs a tree-based strategy to systematically explore and refine response variations, optimizing output quality through multiple iterations. 

This approach is inspired by recent advancements in decision-making algorithms as detailed in [this academic paper](https://arxiv.org/pdf/2406.07394), with a specific adaptation for creative contexts such as short story generation using only local models.

## Applications (current/future)
The main application of this project is to facilitate fine-tuning processes through Direct Preference Optimization (DPO) or similar techniques. The adaptability of the MCTS framework allows for potential expansions into other specialized domains including but not limited to:
- **Mathematical Reasoning**: Currently implemented in a preliminary form, we still need to refine this aspect to enhance logical and problem-solving outputs.
- **Translation**: Exploring the feasibility of applying MCTS for improving translation accuracy and context relevance. [Not implemented yet]
- **Domain-Specific Knowledge**: Tailoring responses to fit specialized knowledge domains more accurately. [Not implemented yet]

### Features
- MCTS algorithm for response refinement using CoT prompts.
- Set a minimal number of depth=1 nodes to force exploration early on (useful to avoid getting stuck into a local optimum with the LLM responses)
- Integration with language models for text generation and evaluation. Supports Alpaca, Llama3, Vicuna, ChatML instruction sets.
- Using advanced samplers like dynamic temperature, minP... (thanks to Ooba/Kobold server) is possible while prompting language models.
- Multiple evaluation metrics including perplexity, readability, coherence, diversity... and LLM auto-evaluation. Each of them can be used alone or combined.
- Importance sampling for efficient node exploration (optional).
- Visualization of the MCTS tree and Q-values at each iteration.
- State saving (JSON) and loading for interruption and resumption of long-running processes, also save the final state into a JSON file (with all the feedbacks/refined answers from all iterations).

The system has been tested with 7B/8B llama models mainly (within Kaggle limitations). It should work seamlessly with bigger models, but models under 7B such as TinyLlama struggle a lot with the complex prompting.

### Visualization example
![Example](https://github.com/AdamCodd/MCTS-LLM/blob/main/example-visualization.png)

### Requirements
- Python 3.7+
- PyTorch
- Transformers
- NetworkX
- Matplotlib
- Spacy
- NLTK
- TextBlob
- Sentence-Transformers

This project is still a WIP but perfectly usable in its current state. Also there is only a Jupyter notebook to run the project as I'm currently running the project with Kaggle.

If you want to support me, you can [here](https://ko-fi.com/adamcodd). You can find all my finetuned NLP/vision models on [HuggingFace](https://huggingface.co/AdamCodd).

License: Creative Commons Attribution-NonCommercial-ShareAlike (CC-BY-NC-SA).

###  Citation and Acknowledgments
```bibtex
@misc{zhang2024accessing,
      title={Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B}, 
      author={Di Zhang and Xiaoshui Huang and Dongzhan Zhou and Yuqiang Li and Wanli Ouyang},
      year={2024},
      eprint={2406.07394},
      archivePrefix={arXiv},
      primaryClass={id='cs.AI' full_name='Artificial Intelligence' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers all areas of AI except Vision, Robotics, Machine Learning, Multiagent Systems, and Computation and Language (Natural Language Processing), which have separate subject areas. In particular, includes Expert Systems, Theorem Proving (although this may overlap with Logic in Computer Science), Knowledge Representation, Planning, and Uncertainty in AI. Roughly includes material in ACM Subject Classes I.2.0, I.2.1, I.2.3, I.2.4, I.2.8, and I.2.11.'}
}
```
