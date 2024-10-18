# Watch Every Step! LLM Agent Learning via Iterative Step-level Process Refinement

This repository contains the code for the paper "Watch Every Step! LLM Agent Learning via Iterative Step-level Process Refinement"

<p align="center">
<img src=assets/main.png width=700/>
</p>

In this work, we introduce the **I**terative step-level **P**rocess **R**efinement (IPR) framework, designed to enhance the training process of Large Language Model (LLM) agents through detailed step-by-step guidance. The IPR framework utilizes the Monte Carlo method to estimate step-level rewards and, during each iteration, allows the agent to explore and generate new actions along the expert trajectory. These actions are then evaluated against the corresponding step of the expert trajectory using step-level rewards to identify discrepancies, creating contrastive action pairs that serve as training data for the agent.

## 🧩 Structure of This Project
There are three main folders in this project: `envs`, `eval_agent`, `fastchat`, `textworld`

`envs`: the interaction environment of WebShop. We transform the original [WebShop](https://github.com/princeton-nlp/WebShop) repo into a package.

`eval_agent`: the evaluation framework of agent tasks, which is inspired by [MINT](https://github.com/xingyaoww/mint-bench).

`fastchat`: training scripts for SFT and DPO, which is a modified version of [FastChat](https://github.com/lm-sys/FastChat).

`textworld`: A text-based game generator and extensible sandbox learning environment for training and testing reinforcement learning (RL) agents.

## 🛠️ Setup

```bash
bash setup.sh
```

The setup script performs the following actions:

- Install Python dependencies for agent training, deployment, evaluation, and the environments for WebShop, InterCodeSQL, ALFWorld
- Download data and search engine indices for WebShop
- Download game files for ALFWorld
- Download data and create mysql environment for InterCodeSQL
- Download expert trajectories for supervised fine-tuning and mixture trajectory optimization

## 🚀 Quick Start

First, launch the controller of FastChat
```bash
python -m fastchat.serve.controller
```

The bash script `run_pipeline.sh` implements the IPR pipeline. For example, you can run:
```bash
bash run_pipeline.sh
```

The script performs the pipeline of IPR:

1. SFT phase: using the expert trajectories to conduct SFT to get the base agent
2. Evaluate SFT agent
   1. Launch the FastChat controller
   2. Launch the FastChat model worker
   3. Run the evaluation
   4. Kill the model worker. The controller will be reused in the following steps
3. Launch multiple FastChat model workers and let the base agent to explore the environment in parallel
4. Estimate step-level rewards
5. Build contrastive action pairs
6. Conduct mixture trajectory optimization to learn from incorrect actions
7. Evaluate IPR agent
8. Repeat 3-7 to iteratively update the policy

## 🎮 Evaluation

First, launch the controller of FastChat
```bash
python -m fastchat.serve.controller
```

Then, launch the model worker of FastChat
```bash
python -m fastchat.serve.model_worker --model-path <YOUR_MODEL_PATH> --port 21002 --worker-address http://localhost:21002
```

Finally, evaluate the agent
```bash
python -m eval_agent.main --agent_config fastchat --model_name <YOUR_MODEL_NAME> --exp_config <TASK_NAME> --split test --verbose
```

## 🔁 vLLM Integration

You can use vLLM as an optimized worker implementation in FastChat. It offers advanced continuous batching and a much higher (~10x) throughput. 

First, install vLLM.
```
pip install vllm==0.4.0.post1
```

However, the version of vLLM may conflict with the version of PyTorch and transformers. It is necessary to reinstall both the transformers and PyTorch libraries.

```
pip install transformers==4.37.2
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```


The terminal would display a version conflict between vLLM, PyTorch, and transformers libraries. You can ignore this warning and proceed to run the `run_pipeline_vllm.sh` script.

```
bash run_pipeline_vllm.sh
```


## 📑 The Data Format for Training the Agent

### SFT data

```json
[
  {
    "id": "example_0",
    "conversations": [
            {
                "from": "human",
                "value": "You are web shopping.\nI will give you .."
            },
            {
                "from": "gpt",
                "value": "OK"
            },
            {
                "from": "human",
                "value": "WebShop [SEP] Instruction: [SEP] i would like a bundle of crackers, spicy beef and cheese which is shelf stable. it also needs to be keto and gluten free, and price lower than 40.00 dollars [SEP] Search"
            },
            {
                "from": "gpt",
                "value": "Thought: I need to find a bundle that includes crackers, spicy beef, and cheese, which is shelf stable, keto, gluten-free, and under $40.00. I'll start by searching for a general bundle of these items.\nAction: search[bundle of crackers, spicy beef and cheese]"
            },
            {
                "from": "human",
                "value": "Observation: [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B0978Q1KK9 [SEP] The Spicy Beef ..."
            },
            {
                "from": "gpt",
                "value": "Thought: The search results show several options. The Spicy Beef Backpacking Bundle (B0978Q1KK9) seems to match my criteria and is well within my budget. I'll click on it for more details.\nAction: click[b0978q1kk9]"
            },
            {
                "from": "human",
                "value": "Observation: [SEP] Back to Search [SEP] < Prev [SEP] flavor name [SEP] original beef backpack bundle [SEP] spicy beef ..."
            },
            {
                "from": "gpt",
                "value": "Thought: Now that I've clicked on the Spicy Beef Backpacking Bundle, I need to confirm that it's the spicy beef and cheese flavor, shelf stable, keto, and gluten-free as required.\nAction: click[spicy beef backpack bundle]"
            },
            {
                "from": "human",
                "value": "Observation: [SEP] Back to Search [SEP] < Prev [SEP] flavor name [SEP] original beef backpack bundle [SEP] spicy beef ..."
            },
            {
                "from": "gpt",
                "value": "Thought: The bundle has the correct flavor and is priced at $14.49, which is affordable. It doesn't explicitly say if it's keto and gluten-free, but since it meets the other criteria and is within budget, I'll proceed with the purchase.\nAction: click[buy now]"
            }
        ]
  }
]
```

### IPR data

```json
[
    {
        "id": "example_0",
        "prompt": [
            {
                "from": "human",
                "value": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round ..."
            },
            {
                "from": "gpt",
                "value": "OK"
            },
            {
                "from": "human",
                "value": "WebShop [SEP] Instruction: [SEP] i want a yellow easy to carry gaone fm radio alarm clock, and price lower than 70.00 dollars [SEP] Search"
            },
            {
                "from": "gpt",
                "value": "Thought: I need to find a yellow Gaone FM radio alarm clock that is easy to carry and costs less than $70. I'll start by searching for \"gaone radio alarm clock yellow\" to see the available options.\nAction: search[gaone radio alarm clock yellow]"
            },
            {
                "from": "human",
                "value": "Observation: [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B09GK7V42L [SEP] Gaone FM Radio ..."
            },
            {
                "from": "gpt",
                "value": "Thought: There are several options, but B09GK7V42L is a yellow Gaone FM radio alarm clock within my budget. I should click on it to get more details.\nAction: click[b09gk7v42l]"
            },
            {
                "from": "human",
                "value": "Observation: [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] black [SEP] white [SEP] yellow [SEP] Gaone FM Radio Alarm ..."
            }
        ],
        "chosen": [
            {
                "from": "gpt",
                "value": "Thought: This radio alarm clock is indeed yellow and costs $58.78, which is under my $70 limit. It also has the features I'm looking for, such as voice control and adjustable brightness. I should select the yellow color to confirm my choice.\nAction: click[yellow]"
            },
            {
                "from": "human",
                "value": "Observation: [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] black [SEP] white [SEP] yellow [SEP] Gaone FM Radio ..."
            },
            {
                "from": "gpt",
                "value": "Thought: I've confirmed the color, and the price is still within my budget. It's time to finalize my purchase by clicking the 'Buy Now' button.\nAction: click[buy now]"
            }
        ],
        "rejected": [
            {
                "from": "gpt",
                "value": "Thought: The product details confirm it's yellow and the price is $58.78, which is under my $90 limit. It seems to meet all my criteria, so I'll go ahead and buy it.\nAction: click[buy now]"
            }
        ]
    }
]
```

## 📖 Citation

If you find this repo helpful, please cite out paper:

```
@article{xiong2024watch,
  title={Watch Every Step! LLM Agent Learning via Iterative Step-Level Process Refinement},
  author={Xiong, Weimin and Song, Yifan and Zhao, Xiutian and Wu, Wenhao and Wang, Xun and Wang, Ke and Li, Cheng and Peng, Wei and Li, Sujian},
  journal={arXiv preprint arXiv:2406.11176},
  year={2024}
}
```
