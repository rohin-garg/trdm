Diffusion Harness Walkthrough
==============================

1. TRM latent layout (L-level = 916 tokens)
------------------------------------------

The Tiny Recursive Model (TRM) adds a small puzzle embedding in front of the 900 ARC tokens. In `models/recursive_reasoning/trm.py` the inner model widens the input length by `puzzle_emb_len`:

```
self.puzzle_emb_len = ...  # default 16
embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)
```

With `seq_len = 900` from `datasets/*/train/dataset.json` and `puzzle_emb_len = 16`, every latent slice (`x`, `z_L`, `z_H`) has length `900 + 16 = 916` and width `hidden_size = 512`. The carry tensors are allocated with that length:

```
TinyRecursiveReasoningModel_ACTV1_Inner.empty_carry(...):
    z_H: [batch, 916, 512]
    z_L: [batch, 916, 512]
```

Inside the forward pass (`store_diffusion_trajectory=True`) the outer loop runs over H-cycles and the inner loop runs over L-cycles:

```
for _H_step in range(self.config.H_cycles):  # default 3
    z_L_list = []
    for _L_step in range(self.config.L_cycles):  # default 6
        z_L = self.L_level(z_L, z_H + input_embeddings, ...)
        z_L_list.append(z_L.clone())
    z_H = self.L_level(z_H, z_L, ...)
    all_h_cycles.append({
        "z_L_list": z_L_list,        # 6 tensors, each [batch, 916, 512]
        "z_H": z_H.clone(),          # [batch, 916, 512]
        "input_embeddings": input_embeddings.clone()
    })
```

The harness therefore records **7** latents per H-cycle: six `z_L` updates plus the terminal `z_H` update. These are the primitives that the diffusion jobs learn to predict.

2. Trajectory capture and streaming
-----------------------------------

With ~1M training examples, storing all tracjetories would take up 1e6 * 16 * 3 * 7  * 916 * 512 * 2 bytes = 300 TB in FP32, which is too much. We hence stream inference during training.

`diffusion_streaming.py` wrap the frozen TRM so we can harvest trajectories without touching storage. `_BaseTrajectoryStreamer` reconstructs the pretraining dataloader, runs the TRM forward pass, and keeps everything on GPU until the copy to CPU tensors at the end of `_next_trajectory_batch`:

```
carry, outputs = model(
    carry=carry,
    batch=device_batch,
    store_diffusion_trajectory=True,
)
trajectory = outputs["trajectory"]
```

Each batch becomes a dictionary with the structure described above; the helper methods reshape it into per-sample examples:

- `LayerUpdateStreamer` flattens every `z_L` transition plus the `z_H` update into individual training rows. For each sample it yields:

  * `inputs`: tuple of `[916, 512]` tensors (`(z_prev, z_H_prev, x)` for L-updates or `(z_H_prev, z_last)` for the final Y update).
  * `target`: the next latent `[916, 512]`.
  * `type`: 0 for `z_L`, 1 for `z_H`.

- `HCycleStreamer` stacks the **entire** H-cycle into a single `[7, 916, 512]` tensor (`torch.stack([...])`) and stores the conditioning tensors separately (`x`, `y_init`, `z_init`).

The collate helpers keep the shapes explicit:

```
collate_layer_examples -> ((inputs...), targets, types)
collate_hcycle_examples -> ((x, y_init, z_init), targets)
```

This is the only place where we reshape data before it reaches the diffusion models.

3. Diffusion-916 (single L-update harness)
-----------------------------------------

`models/diffusion_916.py` keeps the TRM micro-architecture—attention blocks plus MLP—to predict the noise on a single `[916, 512]` latent. The forward pass shows every conditioning tensor and confirms they are summed element-wise:

```
def forward(self, noisy_target, timestep, *condition_inputs):
    condition = sum(condition_inputs)             # each [B, 916, 512]
    time_emb = self.time_mlp(timestep_embedding)  # broadcasts to [B, 1, 512]
    hidden = rms_norm(noisy_target + condition + time_emb)
    for layer in self.layers:
        hidden = layer(hidden, cos_sin)
    return hidden  # predicted noise, still [B, 916, 512]
```

`DiffusionTrainer916.train_step` injects the standard DDPM schedule and computes the noise target:

```
noise = torch.randn_like(target)
alpha_t, sigma_t = schedule.sqrt_alphas_cumprod[t], schedule.sqrt_one_minus_alphas_cumprod[t]
noisy_latent = alpha_t * target + sigma_t * noise
pred_eps = self.model(noisy_latent, t, *condition_inputs)
loss = F.mse_loss(pred_eps, noise)
```

Because we flatten every transition, each TRM trajectory of three H-cycles produces `3 × 7 = 21` training rows at this level.

4. Diffusion-9160 (full H-cycle harness)
----------------------------------------

`models/diffusion_9160.py` learns to denoise the whole H-cycle at once. The number `9160` comes directly from concatenating the three conditioning blocks plus the seven latent steps:

```
num_trajectory_steps = 7
full_input = torch.cat([
    cond_x.unsqueeze(1),           # x      -> 1 × 916 tokens
    cond_y.unsqueeze(1),           # y_init -> 1 × 916 tokens
    cond_z.unsqueeze(1),           # z_init -> 1 × 916 tokens
    noisy_with_pos                 # 7 × 916 tokens (z1..z6, y_next)
], dim=1)
full_input_flat = full_input.view(B, -1, hidden_dim)
# full_input_flat.shape == [B, 10 * 916, 512] -> [B, 9160, 512]
```

After the Transformer encoder the head discards the first `3 × 916` tokens and projects the remaining `7 × 916` locations back to latent space, so the model always outputs `[B, 6412, 512]` (`7 × 916`) noise predictions that align with the stacked targets from `HCycleStreamer`.

5. Relationship between the two levels
--------------------------------------

- L-level (916 tokens): learner sees one TRM update at a time, conditioned on the previous latent(s). Training data size grows with `H_cycles × (L_cycles + 1)` because we break the trajectory into independent examples.
- H-level (9160 tokens): learner sees all updates in the H-cycle jointly, conditioned on `(x, y_init, z_init)`. The sequence length expands to 10 blocks of 916 tokens (`3` conditioners + `7` predicted steps).

Both harnesses derive from the identical trajectory capture path, so switching between them only changes the streaming adapter and the diffusion head that consumes the tensors.

