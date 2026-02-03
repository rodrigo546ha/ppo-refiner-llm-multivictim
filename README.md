# LLM Red-Teaming via PPO-Based Prompt Refinement

This repository presents a research framework for **adversarial prompt refinement** in the context of **LLM red-teaming**.  
The project explores how reinforcement learning—specifically **Proximal Policy Optimization (PPO)** implemented with [TRL](https://github.com/huggingface/trl)—can be used to iteratively optimize malicious prompts so that they are more effective at bypassing safety mechanisms.

---

## Overview

The system is built around three main components:

1. **Refiner** — a PPO-trained model that generates candidate prompts.  
2. **Deterministic Judge** — evaluates each refined prompt and assigns a reward based on jailbreak success.  
3. **Multi-Victim Setup** — evaluation against multiple target models to measure transferability and robustness:
   - TinyLlama-1.1B-Chat
   - Gemma-2B-IT
   - Qwen2.5-3B-Instruct

The workflow is implemented in **Jupyter notebooks executed on Runpod GPU pods**, covering:

- **Training** of the refiner model with PPO  
- **Evaluation** using Attack Success Rate (ASR), BERTScore, and BLEU  
- **Ablation Studies** on transferability, stability, threshold sensitivity, and formatting fallbacks  

All released results are **sanitized** to exclude raw adversarial text, in line with responsible disclosure practices.

---

## Repository Structure

```text
.
├── README.md                # this file
├── notebooks/
│   ├── 01_train.ipynb       # PPO training loop
│   ├── 02_eval.ipynb        # evaluation (ASR, BERTScore, BLEU)
│   ├── 03_ablations.ipynb   # ablation studies (A1–A5)
├── results/
│   ├── eval/                # evaluation CSVs
│   ├── ablations/           # ablation results (CSV)



git clone https://github.com/rodrigo546ha/ppo-refiner-llm-multivictim.git
cd ppo-refiner-llm-multivictim
pip install -r requirements.txt

---
## Results:
Key metrics used in this work:

ASR (Attack Success Rate) — proportion of prompts that bypass defenses.

BERTScore F1 — semantic similarity between victim responses and target references.

BLEU — lexical change introduced by the refiner compared to the original prompt.

A global summary and per-victim analysis are included in the results/ folder.

---

## Citation:
@misc{llm_redteam_refiner,
  title        = {PPO Refiner for Multi-Victim LLM Red-Teaming},
  author       = {Rodrigo},
  year         = {2025},
  howpublished = {\url{https://github.com/rodrigo546ha/ppo-refiner-llm-multivictim}}
}
