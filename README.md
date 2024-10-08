# Towards Implicit Bias Detection and Mitigation in Multi-Agent LLM Interactions

Current LLMs use RLHF to reduce explicit bias in their outputs. But do they also address *implicit* bias? 

In our *EMNLP 2024 (Findings)* [paper](https://arxiv.org/pdf/2410.02584), we identify the presence of implicit bias in multi-agent LLM interactions and propose strategies to address these biases. 

![image](https://github.com/user-attachments/assets/74fb07ee-af60-44d5-bf8f-4ff95da33153) 


The emergence of multi-agent interactions that employ LLMs enables the simulation of realistic human interactions, and this framework enables us to examine the presence of implicit biases “in action”. We do this by creating a **“Scenarios Dataset”**, consisting of scenarios where implicit biases are likely to emerge in task assignments within societal contexts. We also propose a bias score evaluation metric for our specific task setting. 

<p align="left">
  <img src="https://github.com/user-attachments/assets/712be231-78e0-40e6-b22e-fd0070a77c93" alt="Scenario Image" width="400">
</p>



We find that biases increase after multi-agent interaction. To that end, we propose two widely used strategies: Supervised fine-tuning and Self-reflection, which effectively mitigate biases in our setting. For more information, read our paper: 

[Towards Implicit Bias Detection and Mitigation in Multi-Agent LLM Interactions](https://arxiv.org/pdf/2410.02584) 

By [Angana Borah](https://anganab.github.io/) and [Rada Mihalcea](https://web.eecs.umich.edu/~mihalcea/)


## Lessons Learned 
1. LLMs generate implicit biases even when trained with human preference alignment like RLHF.
2. Larger models are prone to produce more biased outputs.
3. Biases increase after multi-agent LLM interactions.
4. Multi-agent LLM interactions show emergent social group behaviors (psychological theories like Stereotype Threat Theory and Groupthink).

## Data and Code 

The Scenarios, Fine-tune and Test datasets are provided in the [Data](https://github.com/MichiganNLP/MultiAgent_ImplicitBias/tree/main/Data) folder. 

The codebase for the multi-agent framework is in the [Code](https://github.com/MichiganNLP/MultiAgent_ImplicitBias/tree/main/Code) folder. 

## Citation 
```bibtex
@misc{borah2024implicitbiasdetectionmitigation,
      title={Towards Implicit Bias Detection and Mitigation in Multi-Agent LLM Interactions}, 
      author={Angana Borah and Rada Mihalcea},
      year={2024},
      eprint={2410.02584},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.02584}, 
}
