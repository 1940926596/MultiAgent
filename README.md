# 🧠 LLM-Driven Financial Multi-Agent System

This project implements a local multi-agent financial decision-making system powered by LLaMA3. It combines large language models (LLMs), reinforcement learning, and role-specialized agents to simulate human-like trading strategies. The system integrates multiple LLM agents with vectorized coordination and a PPO-based policy network to generate optimized trading actions in dynamic markets.

## 🚀 Project Overview

- **Agents**: 5 domain-specific LLM agents (Technical Analyst, Fundamental Analyst, Sentiment Analyst, Risk Controller, CIO).
- **Model**: Locally deployed LLaMA3 with standardized ReAct prompting, function calling, and RAG-based knowledge retrieval.
- **Fusion**: Vectorized agent outputs are weighted by a learnable `AgentWeightPolicyNet` (softmax-based).
- **Final Decision**: A PPO-trained `PolicyNet` integrates fused agent outputs and market state to make trading decisions.
- **Outputs**: For each timestep: `action ∈ {Buy, Hold, Sell}`, `confidence score`, `reasoning`.

## 🧩 Architecture

```text
           ┌──────────────┐
           │  LLM Agent x5│ (TA, FA, Sentiment, Risk, CIO)
           └──────┬───────┘
                  │
     [Buy/Hold/Sell + Confidence]
                  ↓
    ┌────────────────────────────┐
    │  AgentWeightPolicyNet      │ ← Supervised Training
    │ (input: agent decisions)   │
    │ (output: fusion weights)   │
    └────────────┬───────────────┘
                 ↓
 [Weighted Fusion Vector + Market State]
                 ↓
        ┌────────────────────┐
        │     PolicyNet      │ ← PPO Training
        │ (outputs final action) │
        └────────────────────┘
