from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor
    # NOTE: Additional fields may be added by streaming utilities; keep minimal here.


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.register_buffer(
            "H_init",
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.register_buffer(
            "L_init",
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor], store_diffusion_trajectory: bool = False) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[Dict]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Trajectory collection
        trajectory = None
        if store_diffusion_trajectory:
            all_h_cycles = []

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        
        # Collect trajectories for ALL H_cycles
        for _H_step in range(self.config.H_cycles):
            if store_diffusion_trajectory:
                z_L_list = []
            
            # L_cycles
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                if store_diffusion_trajectory:
                    z_L_list.append(z_L.clone())
            
            # H update
            z_H = self.L_level(z_H, z_L, **seq_info)
            
            if store_diffusion_trajectory:
                all_h_cycles.append({
                    'z_L_list': z_L_list,  # List of 6 tensors, each [B, 916, 512]
                    'z_H': z_H.clone(),     # [B, 916, 512]
                    'input_embeddings': input_embeddings.clone()  # [B, 916, 512]
                })
        
        if store_diffusion_trajectory:
            trajectory = {
                'all_h_cycles': all_h_cycles,  # List of 3 H-cycles
                'initial_z_H': carry.z_H.clone(),
                'initial_z_L': carry.z_L.clone()
            }

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), trajectory

    @torch.no_grad()
    def streaming_forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        steps: int,
        noise_scale: float = 0.0,
        collect_core_io: bool = False,
        core_idxs: Optional[List[int]] = None,
        replacements: Optional[Dict[int, callable]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Halting-free forward pass that decouples every TRM core call.

        One halting step consists of H_cycles iterations; each H_cycle performs
        L_cycles updates to z_L followed by one update to z_H. With the default
        config (H_cycles=3, L_cycles=6) this is 21 cores per step, so `steps=4`
        yields 84 distinct core indices. We inject Gaussian noise on the
        post-injection activations (hidden + injection) before the shared
        L-level block. Noise scale is a scalar derived from the RMS of the
        entire core input (flattened). If `replacements` provides a callable for
        a core_idx, that output is used instead. When `collect_core_io=True`,
        records contain (core_idx, x_clean, noise, core_output) for the requested
        core_idxs (defaults to [83], the final core in a 4-step run).
        """
        device = batch["inputs"].device
        inputs = batch["inputs"]
        puzzle_ids = batch["puzzle_identifiers"]
        input_embeddings = self._input_embeddings(inputs, puzzle_ids)

        # Init states
        B = inputs.shape[0]
        total_seq = input_embeddings.shape[1]
        z_H = self.H_init.to(device).expand(B, total_seq, -1)
        z_L = self.L_init.to(device).expand(B, total_seq, -1)

        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        records: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        core_idx = 0
        core_idx_set = None
        if collect_core_io:
            target_core_idxs = list(core_idxs) if core_idxs is not None else [83]
            core_idx_set = set(target_core_idxs)

        def _run_core(hidden_states: torch.Tensor, injection: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            nonlocal core_idx
            x_clean = hidden_states + injection
            x_clean = x_clean.to(self.forward_dtype)

            if noise_scale > 0:
                # Per-sample scalar RMS over the full core input (flattened).
                rms = torch.sqrt(torch.mean(x_clean.to(torch.float32) ** 2, dim=(1, 2)) + 1e-6)  # [B]
                noise = torch.randn_like(x_clean) * (noise_scale * rms.view(-1, 1, 1))
                x_noisy = x_clean + noise
            else:
                noise = torch.zeros_like(x_clean)
                x_noisy = x_clean

            if replacements is not None and core_idx in replacements:
                out = replacements[core_idx](x_clean, noise)
            else:
                hidden_states = x_noisy
                for layer in self.L_level.layers:
                    hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
                out = hidden_states

            if core_idx_set is not None and core_idx in core_idx_set:
                # Store core_idx, clean input, noise used, and output.
                records.append((core_idx, x_clean.detach(), noise.detach(), out.detach()))

            core_idx += 1
            return out, None

        # Fixed halting steps; ignore ACT. Each halting step runs H_cycles blocks,
        # each of which runs L_cycles updates to z_L then one update to z_H.
        for _step in range(steps):
            for _h in range(self.config.H_cycles):
                for _l in range(self.config.L_cycles):
                    z_L, _ = _run_core(z_L, z_H + input_embeddings)
                z_H, _ = _run_core(z_H, z_L)

        logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        return logits, records

    def streaming_forward_for_rl(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        steps: int,
        noise_scale: float = 0.0,
        collect_core_io: bool = False,
        core_idxs: Optional[List[int]] = None, # this is the list of cores that we're injecting noise into
        replacements: Optional[Dict[int, callable]] = None,
        fixed_noise_seeds: Optional[Dict[int, torch.Tensor]] = None,
        expand_fixed_noise: Optional[int] = None, # must the fixed noise be expanded by this factor?
        post_process_noise_scale: Optional[float] = None, # do I add post processed noise to the things in the core_idxs
    ) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Halting-free forward pass that decouples every TRM core call.

        One halting step consists of H_cycles iterations; each H_cycle performs
        L_cycles updates to z_L followed by one update to z_H. With the default
        config (H_cycles=3, L_cycles=6) this is 21 cores per step, so `steps=4`
        yields 84 distinct core indices. We inject Gaussian noise on the
        post-injection activations (hidden + injection) before the shared
        L-level block. Noise scale is a scalar derived from the RMS of the
        entire core input (flattened). If `replacements` provides a callable for
        a core_idx, that output is used instead. When `collect_core_io=True`,
        records contain (core_idx, x_clean, noise, core_output) for the requested
        core_idxs (defaults to [83], the final core in a 4-step run).
        """
        device = batch["inputs"].device
        inputs = batch["inputs"]
        puzzle_ids = batch["puzzle_identifiers"]
        input_embeddings = self._input_embeddings(inputs, puzzle_ids)

        # Init states
        B = inputs.shape[0]
        total_seq = input_embeddings.shape[1]
        z_H = self.H_init.to(device).expand(B, total_seq, -1)
        z_L = self.L_init.to(device).expand(B, total_seq, -1)

        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        records: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        core_idx = 0
        core_idx_set = None
        if collect_core_io:
            target_core_idxs = list(core_idxs) if core_idxs is not None else [83]
            core_idx_set = set(target_core_idxs)

        def _run_core(hidden_states: torch.Tensor, injection: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            nonlocal core_idx
            x_clean = hidden_states + injection
            x_clean = x_clean.to(self.forward_dtype)

            if noise_scale > 0:
                # Per-sample scalar RMS over the full core input (flattened).
                rms = torch.sqrt(torch.mean(x_clean.to(torch.float32) ** 2, dim=(1, 2)) + 1e-6)  # [B]
                if fixed_noise_seeds is not None and core_idx in fixed_noise_seeds:
                    # if the first thing is [B*G, ...], then we want to only sample [B, ...] and then expand the first dimension by G
                    wanted_shape = list(x_clean.shape)
                    if expand_fixed_noise is not None:
                        wanted_shape[0] //= expand_fixed_noise
                    # print(f"wanted_shape: {wanted_shape}")
                    noise = torch.empty(*wanted_shape, dtype=x_clean.dtype, device=x_clean.device).normal_(generator=torch.Generator(device=x_clean.device).manual_seed(fixed_noise_seeds[core_idx].item()))
                    if expand_fixed_noise is not None:
                        noise = noise.repeat_interleave(expand_fixed_noise, dim=0)
                    assert noise.shape == x_clean.shape
                    # noise = torch.empty_like(x_clean).normal_(generator=torch.Generator(device=x_clean.device).manual_seed(fixed_noise_seeds[core_idx].item()))
                else:
                    noise = torch.randn_like(x_clean)
                noise = noise * (noise_scale * rms.view(-1, 1, 1))
                x_noisy = x_clean + noise
            else:
                noise = torch.zeros_like(x_clean)
                x_noisy = x_clean

            if replacements is not None and core_idx in replacements:
                out = replacements[core_idx](x_clean, noise, core_idx)
            else:
                hidden_states = x_noisy
                for layer in self.L_level.layers:
                    hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
                out = hidden_states

            if core_idx_set is not None and core_idx in core_idx_set:
                # Store core_idx, clean input, **POST PROCESSED** noise used, and output.
                if post_process_noise_scale is not None:
                    rms = torch.sqrt(torch.mean(out.to(torch.float32) ** 2, dim=(1, 2)) + 1e-6)
                    post_processed_noise = torch.randn_like(noise) * (post_process_noise_scale * rms.view(-1, 1, 1))
                    out += post_processed_noise
                    records.append((core_idx, x_clean, post_processed_noise, out))
                else:
                    assert core_idx is not None
                    assert x_clean is not None
                    assert out is not None
                    records.append((core_idx, x_clean, None, out))
                

            core_idx += 1
            return out, None

        # Fixed halting steps; ignore ACT. Each halting step runs H_cycles blocks,
        # each of which runs L_cycles updates to z_L then one update to z_H.
        for _step in range(steps):
            for _h in range(self.config.H_cycles):
                for _l in range(self.config.L_cycles):
                    z_L, _ = _run_core(z_L, z_H + input_embeddings)
                z_H, _ = _run_core(z_H, z_L)

        logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        return logits, records


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
        *,
        return_latents: bool = False,
        store_diffusion_trajectory: bool = False,
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), trajectory = self.inner(new_inner_carry, new_current_data, store_diffusion_trajectory)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        if return_latents:
            outputs["z_H"] = new_inner_carry.z_H
            outputs["z_L"] = new_inner_carry.z_L
        
        if store_diffusion_trajectory:
            outputs["trajectory"] = trajectory

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs

    @torch.no_grad()
    def streaming_forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        steps: int,
        noise_scale: float = 0.0,
        collect_core_io: bool = False,
        core_idxs: Optional[List[int]] = None,
        replacements: Optional[Dict[int, callable]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Convenience wrapper around inner.streaming_forward.
        """
        return self.inner.streaming_forward(
            batch=batch,
            steps=steps,
            noise_scale=noise_scale,
            collect_core_io=collect_core_io,
            core_idxs=core_idxs,
            replacements=replacements,
        )

    def streaming_forward_for_rl(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        steps: int,
        noise_scale: float = 0.0,
        collect_core_io: bool = False,
        core_idxs: Optional[List[int]] = None,
        replacements: Optional[Dict[int, callable]] = None,
        fixed_noise_seeds: Optional[Dict[int, torch.Tensor]] = None,
        expand_fixed_noise: Optional[int] = None, # must the fixed noise be expanded by this factor?
        post_process_noise_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]]:
        return self.inner.streaming_forward_for_rl(
            batch=batch,
            steps=steps,
            noise_scale=noise_scale,
            collect_core_io=collect_core_io,
            core_idxs=core_idxs,
            replacements=replacements,
            fixed_noise_seeds=fixed_noise_seeds,
            expand_fixed_noise=expand_fixed_noise,
            post_process_noise_scale=post_process_noise_scale,
        )
