# Tiny Diffusion Recursive Models (TDRM)

## Ideas we're exploring
The goal is to represent a single latent z vector trajectory from a [TRM](https://arxiv.org/pdf/2510.04871) as a diffusion model. The motivation is that by using diffusion to model these continuous reasoning tokens it would improve performance due to the bidirectional nature of synthesizing the entire trace at once (as opposed to autoregressively creating the trace).

Some relevant papers
- [TRM](https://arxiv.org/pdf/2510.04871) ([code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)): a recent result solving small reasoning tasks
- [Diffusion LM](https://arxiv.org/abs/2205.14217): original research on how to do continuous diffusion for language models
- [SPG](https://arxiv.org/pdf/2510.09541): a recent result on policy gradient algorithms for diffusion language models, extending the loss function
- [DeepSeekMath](https://arxiv.org/abs/2402.03300): introduced GRPO and RL on diffusion models

The goal is to eventually surpass TRM's performance on ARC AGI 2. 

## High level research plan
We will start with the sudoku task (for simplicity) and move to ARC AGI 2 later.

1. Distill the standard TRM into TDRM on sudoku
    a. Generate many reasoning traces on data, store in an S3 bucket
    b. Verify that our architecture is as expressive as a TRM and can produce similar quality results 
2. Use RL/GRPO to improve results on sudoku
3. Attempt to train the model conditioned on the correct answer, adding an off-policy loss term to maintain faithfulness
3. Move above to ARC AGI, then experiment on various harnesses to most effectively use few-shot learning