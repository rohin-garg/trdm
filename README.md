# Tiny Diffusion Recursive Models (TDRM)

## Problem
We are attempting to train a small reasoning model to solve the [ARC AGI 2](https://arcprize.org/arc-agi/2/) task. 

## Ideas we're exploring
The goal is to represent a single latent z vector trajectory from a [TRM](https://arxiv.org/pdf/2510.04871) as a diffusion model. The motivation is that by using diffusion to model the reasoning tokens it would improve performance due to the bidirectional nature of synthesizing the entire trace at once (as opposed to autoregressively creating the trace), and their continuous nature makes diffusion a great model.

In particular, we will initialize our model with a competent baseline by distilling from the original TRM paper, and then use GRPO-style RL on the model to improve its performance.

Some relevant papers
- [TRM](https://arxiv.org/pdf/2510.04871) ([code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)): a recent result solving small reasoning tasks
- [Diffusion LM](https://arxiv.org/abs/2205.14217): original research on how to do continuous diffusion for language models
- [SPG](https://arxiv.org/pdf/2510.09541): a recent result on policy gradient algorithms for diffusion language models, extending the loss function
- [DeepSeekMath](https://arxiv.org/abs/2402.03300): introduced GRPO and RL on diffusion models


## High level research plan
We will start with the sudoku task (for simplicity) and move to ARC AGI 2 later if time permits.

1. Distill the standard TRM into TDRM on sudoku
    a. Generate many reasoning traces on data, store in an S3 bucket
    b. Verify that our architecture is as expressive as a TRM and can produce similar quality results 
2. Use RL/GRPO to improve results on sudoku

### If time permits (potentially after project timeline)
3. Attempt to train the model conditioned on the correct answer, adding an off-policy loss term to maintain faithfulness
4. Move above to ARC AGI, then experiment on various harnesses to most effectively use few-shot learning