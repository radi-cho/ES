import jax
import jax.numpy as jnp

from functools import partial

from .llm import LLM
from ..base_model import Model, CommonParams
from ..common import PARAM, MM_PARAM, EMB_PARAM, EXCLUDED, Parameter, MM, TMM, Embedding, Linear, call_submodule


class Qwen2RMSNorm(Model):
    @classmethod
    def _forward(cls, common_params, x, eps=1e-6):
        hidden_states = x
        variance = jnp.mean(hidden_states ** 2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + eps)

        return call_submodule(Parameter, 'weight', common_params) * hidden_states


class Qwen2MLP(Model):
    @classmethod
    def _forward(cls, common_params, x):
        return call_submodule(Linear, 'down_proj', common_params, jax.nn.silu(call_submodule(Linear, 'gate_proj', common_params, x)) * call_submodule(Linear, 'up_proj', common_params, x))

class RWKV6Attention(Model):
    @classmethod
    def _forward(cls, common_params, x, state, length, new_starts, H, S, inner_loop):
        T, C = x.shape

        sx = jnp.concatenate([state[:1], x[:-1, :]], dtype=x.dtype)
        sx = jnp.where(new_starts[:, None], jnp.zeros_like(sx), sx)
        sx = sx - x
        xxx = x + sx * call_submodule(Parameter, 'time_maa_x', common_params)
        xxx = jnp.tanh(call_submodule(TMM, 'time_maa_w1', common_params, xxx)).reshape(T, 5, -1).transpose(1, 0, 2)
        xxx = jax.vmap(lambda x, y: x @ y)(xxx, common_params.params['time_maa_w2']).reshape(5, T, -1) # TODO: Fix
        mr, mk, mv, mw, mg = xxx

        xr = x + sx * (call_submodule(Parameter, 'time_maa_r', common_params) + mr)
        xk = x + sx * (call_submodule(Parameter, 'time_maa_k', common_params) + mk)
        xv = x + sx * (call_submodule(Parameter, 'time_maa_v', common_params) + mv)
        xw = x + sx * (call_submodule(Parameter, 'time_maa_w', common_params) + mw)
        xg = x + sx * (call_submodule(Parameter, 'time_maa_g', common_params) + mg)
        state = state.at[0].set(x[length-1])

        r = jnp.reshape(call_submodule(Linear, 'q_proj', common_params, xr), (T, H, S)) # query_states
        k = jnp.reshape(call_submodule(Linear, 'k_proj', common_params, xk), (T, -1, S)) # key_states
        v = jnp.reshape(call_submodule(Linear, 'v_proj', common_params, xv), (T, -1, S)) # value_states

        w_lora_result = call_submodule(Parameter, 'time_decay', common_params).reshape(1, H, S, 1) + call_submodule(TMM, 'time_decay_w2', common_params, jnp.tanh(call_submodule(TMM, 'time_decay_w1', common_params, xw))).reshape(T, H, S, 1) # decay_states
        
        g = jax.nn.sigmoid(call_submodule(Linear, 'gate', common_params, xg)) # gate_states

        num_kv_reps = r.shape[-2] // k.shape[-2]
        k = jnp.repeat(k, num_kv_reps, axis=-2)
        v = jnp.repeat(v, num_kv_reps, axis=-2)

        log_w = -jnp.exp(w_lora_result) # decay_states_log
        log_w = jnp.clip(log_w, min=-5.0)
        k = k * (1 - jnp.exp(log_w[:, :, :, 0]))

        # scale = r.shape[-1] ** -0.5
        s = jnp.reshape(state[1:, :],(H, S, S))
        state_new, out = inner_loop(r, k, v, log_w, None, s, length, new_starts)
        state = state.at[1:].set(state_new.reshape(S, -1))

        x = out.reshape(T, H*S) * g
        return call_submodule(Linear, 'o_proj', common_params, x), state

class BaseRWKV(LLM):
    @classmethod
    def transform_torch_model(cls, torch_model, dtype=jnp.bfloat16):
        import torch
        import re
        w = torch_model
        keys = list(w.keys())
        for k in keys:
            k_new = k.replace("model.", "").replace("layers.", "blocks.")
            if 'time_' in k_new:
                w[k] = w[k].squeeze()
            # if 'time_maa_w2' in k_new:
                # w[k] = w[k].reshape((-1, w[k].shape[-1]))
            if k_new != k:
                w[k_new] = w[k]
                del w[k]
        return w

    @classmethod
    def get_scan_map(cls, config):
        BS = (0,)
        NS = tuple()
        return {
            'blocks': {
                'input_layernorm': {'weight': BS},
                'mlp': {'down_proj': {'weight': BS}, 'gate_proj': {'weight': BS}, 'up_proj': {'weight': BS}},
                'post_attention_layernorm': {'weight': BS},
                'self_attn': {
                    'gate': {'weight': BS},
                    'k_proj': {'bias': BS, 'weight': BS},
                    'o_proj': {'weight': BS},
                    'q_proj': {'bias': BS, 'weight': BS},
                    'time_decay': BS,
                    'time_decay_w1': BS,
                    'time_decay_w2': BS,
                    'time_maa_g': BS,
                    'time_maa_k': BS,
                    'time_maa_r': BS,
                    'time_maa_v': BS,
                    'time_maa_w': BS,
                    'time_maa_w1': BS,
                    'time_maa_w2': (0, 1),
                    'time_maa_x': BS,
                    'v_proj': {'bias': BS, 'weight': BS}}
            },
            'embed_tokens': {'weight': NS},
            'lm_head': {'weight': NS},
            'norm': {'weight': NS}
        }
        # return {
        #     'blocks': {
        #         'att': {'a0': BS, 'a1': BS, 'a2': BS, 'g1': BS, 'g2': BS, 'k_a': BS, 'k_k': BS, 'key': {'weight': BS},
        #                 'ln_x': {'bias': BS, 'weight': BS}, 'output': {'weight': BS},
        #                 'r_k': BS, # BS EXCEPTION
        #                 'receptance': {'weight': BS},
        #                 'v0': BS, 'v1': BS, 'v2': BS,
        #                 'value': {'weight': BS},
        #                 'w0': BS, 'w1': BS, 'w2': BS, 'x_a': BS, 'x_g': BS, 'x_k': BS, 'x_r': BS, 'x_v': BS, 'x_w': BS},
        #         'ffn': {'key': {'weight': BS}, 'value': {'weight': BS}, 'x_k': BS},
        #         'ln1': {'bias': BS, 'weight': BS}, 'ln2': {'bias': BS, 'weight': BS}},
        #     'emb': {'weight': NS},
        #     'head': {'weight': NS},
        #     'ln0': {'bias': NS, 'weight': NS},
        #     'ln_out': {'bias': NS, 'weight': NS}
        # }

    @classmethod
    def get_es_map(cls, config):
        LORA = MM_PARAM
        FULL = PARAM
        return {
            'blocks': {
                'input_layernorm': {'weight': FULL},
                'mlp': {'down_proj': {'weight': LORA}, 'gate_proj': {'weight': LORA}, 'up_proj': {'weight': LORA}},
                'post_attention_layernorm': {'weight': FULL},
                'self_attn': {
                    'gate': {'weight': LORA},
                    'k_proj': {'bias': FULL, 'weight': LORA},
                    'o_proj': {'weight': LORA},
                    'q_proj': {'bias': FULL, 'weight': LORA},
                    'time_decay': FULL,
                    'time_decay_w1': LORA,
                    'time_decay_w2': LORA,
                    'time_maa_g': FULL,
                    'time_maa_k': FULL,
                    'time_maa_r': FULL,
                    'time_maa_v': FULL,
                    'time_maa_w': FULL,
                    'time_maa_w1': LORA,
                    'time_maa_w2': EXCLUDED, # TODO: FIX
                    'time_maa_x': FULL,
                    'v_proj': {'bias': FULL, 'weight': LORA}}
            },
            'embed_tokens': {'weight': EXCLUDED},
            'lm_head': {'weight': EXCLUDED},
            'norm': {'weight': FULL}
        }
        # return {
        #     'blocks': {
        #         'att': {'a0': FULL, 'a1': LORA, 'a2': LORA, 'g1': LORA, 'g2': LORA, 'k_a': FULL, 'k_k': FULL, 'key': {'weight': LORA},
        #                 'ln_x': {'bias': FULL, 'weight': FULL}, 'output': {'weight': LORA},
        #                 'r_k': FULL, # LORA EXCEPTION
        #                 'receptance': {'weight': LORA},
        #                 'v0': FULL, 'v1': LORA, 'v2': LORA,
        #                 'value': {'weight': LORA},
        #                 'w0': FULL, 'w1': LORA, 'w2': LORA, 'x_a': FULL, 'x_g': FULL, 'x_k': FULL, 'x_r': FULL, 'x_v': FULL, 'x_w': FULL},
        #         'ffn': {'key': {'weight': LORA}, 'value': {'weight': LORA}, 'x_k': FULL},
        #         'ln1': {'bias': FULL, 'weight': FULL}, 'ln2': {'bias': FULL, 'weight': FULL}},
        #     'emb': {'weight': EXCLUDED},
        #     'head': {'weight': EXCLUDED},
        #     'ln0': {'bias': FULL, 'weight': FULL},
        #     'ln_out': {'bias': FULL, 'weight': FULL}
        # }

    @classmethod
    def default_state(cls, params, config):
        n_embd = params['embed_tokens']['weight'].shape[1]
        n_layer = params['blocks']['input_layernorm']['weight'].shape[0]
        head_size = config["head_size"]
        n_head = n_embd // head_size
        return jnp.zeros((n_layer, (1 + head_size), n_embd), dtype=params['embed_tokens']['weight'].dtype)

    @classmethod
    def embed(cls, common_params, tokens):
        # TODO: Make this modifiable
        # return common_params.params['emb']['weight'][tokens.ravel()]
        return common_params.params['embed_tokens']['weight'][tokens.ravel()]
    
    @classmethod
    def outhead(cls, common_params, x):
        # TODO: Make this modifiable
        x = call_submodule(Qwen2RMSNorm, 'norm', common_params, x)
        return x @ common_params.params['lm_head']['weight'].T

    @classmethod
    def inner_loop(cls, r, k, v, w, time_first, s, length, new_starts):
        # q, k, v, gk, _, initial_state
        scale = r.shape[-1] ** -0.5
        w = jnp.exp(w)
        out = jnp.empty_like(r)
        out_s = s

        reset_s = jnp.zeros_like(s)
        for t in range(r.shape[0]):
            s = jax.lax.select(new_starts[t], reset_s, s)
            
            rt = jnp.expand_dims(r[t], 1) * scale
            kt = jnp.expand_dims(k[t], 2)
            vt = jnp.expand_dims(v[t], 1)
            at = kt*vt
            s = jnp.astype(at + w[t] * s, r.dtype)
            out = out.at[t].set((rt @ s).squeeze(1))
            out_s = jax.lax.select(t < length, s, out_s)

        return out_s, out

    @classmethod
    def forward_seq(cls, common_params, x, state, length, new_starts):
        params = common_params.params
        config = common_params.frozen_params
        n_embd = params['embed_tokens']['weight'].shape[1]
        n_layer = params['blocks']['input_layernorm']['weight'].shape[0]
        head_size = config["head_size"]
        n_head = n_embd // head_size

        @partial(jax.checkpoint,
                 policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        def block_loop(x, inputs):
            hidden_states = x
            params_i, es_tree_key_i, state = inputs
            block_i = common_params._replace(
                params=params_i,
                es_tree_key=es_tree_key_i
            )

            residual = hidden_states
            hidden_states = call_submodule(Qwen2RMSNorm, 'input_layernorm', block_i, hidden_states)
            hidden_states, state = call_submodule(RWKV6Attention, 'self_attn', block_i, hidden_states, state, length, new_starts, n_head, head_size, cls.inner_loop)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = call_submodule(Qwen2RMSNorm, 'post_attention_layernorm', block_i, hidden_states)
            hidden_states = call_submodule(Qwen2MLP, 'mlp', block_i, hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states, state

        x, state = jax.lax.scan(block_loop, x, (common_params.params['blocks'], common_params.es_tree_key['blocks'], state))

        return x, state


class FastRWKV(BaseRWKV):
    @classmethod
    def default_state(cls, params, config):
        n_embd = params['embed_tokens']['weight'].shape[1]
        n_layer = params['blocks']['input_layernorm']['weight'].shape[0]
        head_size = config["head_size"]
        n_head = n_embd // head_size
        return jnp.zeros((n_layer, (1 + head_size), n_embd), dtype=params['embed_tokens']['weight'].dtype)
        # return [jnp.zeros(((1 + head_size), n_embd), dtype=params['embed_tokens']['weight'].dtype)] * n_layer

    @classmethod
    def forward_seq(cls, common_params, x, state, length, new_starts):
        params = common_params.params
        config = common_params.frozen_params
        n_embd = params['embed_tokens']['weight'].shape[1]
        n_layer = params['blocks']['input_layernorm']['weight'].shape[0]
        head_size = config["head_size"]
        n_head = n_embd // head_size

        for i in range(n_layer):
            hidden_states = x
            # params_i, es_tree_key_i, state = inputs
            params_i = jax.tree.map(lambda a: a[i], common_params.params['blocks'])
            es_tree_key_i = jax.tree.map(lambda a: a[i], common_params.es_tree_key['blocks'])
            state_i = state[i]
            block_i = common_params._replace(
                params=params_i,
                es_tree_key=es_tree_key_i
            )

            residual = hidden_states
            hidden_states = call_submodule(Qwen2RMSNorm, 'input_layernorm', block_i, hidden_states)
            hidden_states, state_i = call_submodule(RWKV6Attention, 'self_attn', block_i, hidden_states, state_i, length, new_starts, n_head, head_size, cls.inner_loop)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = call_submodule(Qwen2RMSNorm, 'post_attention_layernorm', block_i, hidden_states)
            hidden_states = call_submodule(Qwen2MLP, 'mlp', block_i, hidden_states)
            hidden_states = residual + hidden_states

            x = hidden_states
            state = state.at[i].set(state_i)
            # state[i] = state_i

        return x, state


class Qwen35RMSNorm(Model):
    @classmethod
    def _forward(cls, common_params, x, eps=1e-6):
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + eps)
        return x * (1.0 + call_submodule(Parameter, "weight", common_params))


class Qwen35RMSNormGated(Model):
    @classmethod
    def _forward(cls, common_params, x, gate, eps=1e-6):
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + eps)
        # HF Qwen3.5 gated RMSNorm uses direct multiplicative weight (no +1 offset).
        x = x * call_submodule(Parameter, "weight", common_params)
        return x * jax.nn.silu(gate)


class Qwen35MLP(Model):
    @classmethod
    def _forward(cls, common_params, x):
        gate = jax.nn.silu(call_submodule(Linear, "gate_proj", common_params, x))
        up = call_submodule(Linear, "up_proj", common_params, x)
        return call_submodule(Linear, "down_proj", common_params, gate * up)


class Qwen35LinearAttention(Model):
    @classmethod
    def _forward(cls, common_params, x, conv_state, recurrent_state):
        cfg = common_params.frozen_params

        key_dim = cfg["linear_key_head_dim"] * cfg["linear_num_key_heads"]
        value_dim = cfg["linear_value_head_dim"] * cfg["linear_num_value_heads"]
        head_k_dim = cfg["linear_key_head_dim"]
        head_v_dim = cfg["linear_value_head_dim"]
        num_v_heads = cfg["linear_num_value_heads"]

        mixed_qkv = call_submodule(Linear, "in_proj_qkv", common_params, x)  # [1, conv_dim]
        mixed_qkv = mixed_qkv[0]  # [conv_dim]

        # Single-token causal depthwise conv update.
        conv_common = common_params._replace(
            params=common_params.params["conv1d"],
            es_tree_key=common_params.es_tree_key["conv1d"],
            frozen_params=None,
        )
        conv_w = call_submodule(Parameter, "weight", conv_common)
        conv_w = conv_w[:, 0, :]  # [conv_dim, kernel]
        conv_window = jnp.concatenate([conv_state, mixed_qkv[:, None]], axis=-1)  # [conv_dim, kernel]
        mixed_qkv = jnp.sum(conv_window * conv_w, axis=-1)
        mixed_qkv = jax.nn.silu(mixed_qkv)
        new_conv_state = conv_window[:, 1:]

        query = mixed_qkv[:key_dim].reshape(num_v_heads, head_k_dim)
        key = mixed_qkv[key_dim : (2 * key_dim)].reshape(num_v_heads, head_k_dim)
        value = mixed_qkv[(2 * key_dim) :].reshape(num_v_heads, head_v_dim)

        # Match HF fallback kernel: L2-normalize q/k before recurrent update.
        def l2norm(v, eps=1e-6):
            inv = jax.lax.rsqrt(jnp.sum(v * v, axis=-1, keepdims=True) + eps)
            return v * inv
        query = l2norm(query)
        key = l2norm(key)

        z = call_submodule(Linear, "in_proj_z", common_params, x).reshape(num_v_heads, head_v_dim)
        beta = jax.nn.sigmoid(call_submodule(Linear, "in_proj_b", common_params, x)).reshape(num_v_heads)
        a = call_submodule(Linear, "in_proj_a", common_params, x).reshape(num_v_heads)

        A_log = call_submodule(Parameter, "A_log", common_params).reshape(num_v_heads)
        dt_bias = call_submodule(Parameter, "dt_bias", common_params).reshape(num_v_heads)
        g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus((a + dt_bias).astype(jnp.float32))

        scale = 1.0 / jnp.sqrt(head_k_dim)

        def per_head_step(s, q, k, v, beta_h, g_h):
            s = s * jnp.exp(g_h).astype(s.dtype)
            kv_mem = jnp.sum(s * k[:, None], axis=0)
            delta = (v - kv_mem) * beta_h
            s = s + k[:, None] * delta[None, :]
            out = jnp.sum(s * (q * scale)[:, None], axis=0)
            return s, out

        new_recurrent_state, core = jax.vmap(per_head_step)(
            recurrent_state, query, key, value, beta, g.astype(recurrent_state.dtype)
        )

        core = call_submodule(Qwen35RMSNormGated, "norm", common_params, core, z)
        core = core.reshape(1, value_dim)
        out = call_submodule(Linear, "out_proj", common_params, core)
        return out, new_conv_state, new_recurrent_state


class Qwen35SelfAttention(Model):
    @classmethod
    def _rotate_half(cls, x):
        half = x.shape[-1] // 2
        return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    @classmethod
    def _forward(cls, common_params, x, rope_cos, rope_sin, k_cache, v_cache, pos):
        cfg = common_params.frozen_params
        head_dim = cfg["head_dim"]
        num_heads = cfg["num_attention_heads"]
        num_kv = cfg["num_key_value_heads"]
        num_groups = num_heads // num_kv

        q_proj = call_submodule(Linear, "q_proj", common_params, x).reshape(1, num_heads, head_dim * 2)
        query, gate = jnp.split(q_proj, 2, axis=-1)
        query = call_submodule(Qwen35RMSNorm, "q_norm", common_params, query)

        key = call_submodule(Linear, "k_proj", common_params, x).reshape(1, num_kv, head_dim)
        key = call_submodule(Qwen35RMSNorm, "k_norm", common_params, key)
        value = call_submodule(Linear, "v_proj", common_params, x).reshape(1, num_kv, head_dim)

        rotary_dim = rope_cos.shape[-1]
        q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
        k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
        q_rot_half = cls._rotate_half(q_rot)
        k_rot_half = cls._rotate_half(k_rot)
        query = jnp.concatenate([q_rot * rope_cos + q_rot_half * rope_sin, q_pass], axis=-1)
        key = jnp.concatenate([k_rot * rope_cos + k_rot_half * rope_sin, k_pass], axis=-1)
        new_k_cache = k_cache.at[pos].set(key[0])
        new_v_cache = v_cache.at[pos].set(value[0])

        keys = jnp.repeat(new_k_cache, num_groups, axis=1)  # [cache_len, num_heads, head_dim]
        values = jnp.repeat(new_v_cache, num_groups, axis=1)

        q = query[0]  # [num_heads, head_dim]
        logits = jnp.einsum("hd,shd->hs", q, keys) / jnp.sqrt(head_dim)
        cache_len = k_cache.shape[0]
        valid_mask = jnp.arange(cache_len) <= pos
        logits = jnp.where(valid_mask[None, :], logits, -1e30)
        weights = jax.nn.softmax(logits, axis=-1)
        attn = jnp.einsum("hs,shd->hd", weights, values)[None, ...]
        attn = attn.reshape(1, num_heads * head_dim)
        attn = attn * jax.nn.sigmoid(gate.reshape(1, num_heads * head_dim))
        out = call_submodule(Linear, "o_proj", common_params, attn)
        return out, new_k_cache, new_v_cache


class Qwen35RWKV(LLM):
    @classmethod
    def transform_torch_model(cls, torch_model, dtype=jnp.bfloat16):
        import torch

        w = torch_model
        if isinstance(w, torch.nn.Module):
            w = w.state_dict()

        transformed = {}
        for k, v in w.items():
            if k.startswith("model.language_model.layers."):
                new_k = k.replace("model.language_model.layers.", "blocks.")
                transformed[new_k] = v
            elif k.startswith("model.layers."):
                transformed[k.replace("model.layers.", "blocks.")] = v
            elif k.startswith("model.language_model.embed_tokens."):
                transformed[k.replace("model.language_model.", "")] = v
            elif k.startswith("model.embed_tokens."):
                transformed[k.replace("model.", "")] = v
            elif k.startswith("model.language_model.norm."):
                transformed[k.replace("model.language_model.", "")] = v
            elif k.startswith("model.norm."):
                transformed[k.replace("model.", "")] = v
            elif k.startswith("lm_head."):
                transformed[k] = v
        return transformed

    @classmethod
    def transform_config(cls, config):
        if config is None:
            return {}
        cfg = dict(config)
        cfg["linear_layer_indices"] = [i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention"]
        cfg["full_layer_indices"] = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
        rope_params = cfg.get("rope_parameters", {})
        cfg["rope_theta"] = rope_params.get("rope_theta", 10000000.0)
        cfg["partial_rotary_factor"] = rope_params.get("partial_rotary_factor", 0.25)
        cfg["rms_norm_eps"] = cfg.get("rms_norm_eps", 1e-6)
        cfg["attn_cache_len"] = min(4096, int(cfg.get("max_position_embeddings", 4096)))
        return cfg

    @classmethod
    def get_scan_map(cls, config):
        BS = (0,)
        NS = tuple()
        return {
            "blocks": {
                "input_layernorm": {"weight": BS},
                "post_attention_layernorm": {"weight": BS},
                "mlp": {
                    "down_proj": {"weight": BS},
                    "gate_proj": {"weight": BS},
                    "up_proj": {"weight": BS},
                },
                "linear_attn": {
                    "A_log": BS,
                    "dt_bias": BS,
                    "conv1d": {"weight": BS},
                    "in_proj_a": {"weight": BS},
                    "in_proj_b": {"weight": BS},
                    "in_proj_qkv": {"weight": BS},
                    "in_proj_z": {"weight": BS},
                    "norm": {"weight": BS},
                    "out_proj": {"weight": BS},
                },
                "self_attn": {
                    "k_norm": {"weight": BS},
                    "k_proj": {"weight": BS},
                    "o_proj": {"weight": BS},
                    "q_norm": {"weight": BS},
                    "q_proj": {"weight": BS},
                    "v_proj": {"weight": BS},
                },
            },
            "embed_tokens": {"weight": NS},
            "lm_head": {"weight": NS},
            "norm": {"weight": NS},
        }

    @classmethod
    def get_es_map(cls, config):
        LORA = MM_PARAM
        FULL = PARAM
        return {
            "blocks": {
                "input_layernorm": {"weight": FULL},
                "post_attention_layernorm": {"weight": FULL},
                "mlp": {
                    "down_proj": {"weight": LORA},
                    "gate_proj": {"weight": LORA},
                    "up_proj": {"weight": LORA},
                },
                "linear_attn": {
                    "A_log": FULL,
                    "dt_bias": FULL,
                    "conv1d": {"weight": FULL},
                    "in_proj_a": {"weight": LORA},
                    "in_proj_b": {"weight": LORA},
                    "in_proj_qkv": {"weight": LORA},
                    "in_proj_z": {"weight": LORA},
                    "norm": {"weight": FULL},
                    "out_proj": {"weight": LORA},
                },
                "self_attn": {
                    "k_norm": {"weight": FULL},
                    "k_proj": {"weight": LORA},
                    "o_proj": {"weight": LORA},
                    "q_norm": {"weight": FULL},
                    "q_proj": {"weight": LORA},
                    "v_proj": {"weight": LORA},
                },
            },
            "embed_tokens": {"weight": EXCLUDED},
            "lm_head": {"weight": EXCLUDED},
            "norm": {"weight": FULL},
        }

    @classmethod
    def default_state(cls, params, config):
        n_linear = len(config["linear_layer_indices"])
        n_full = len(config["full_layer_indices"])
        num_v_heads = config["linear_num_value_heads"]
        head_k_dim = config["linear_key_head_dim"]
        head_v_dim = config["linear_value_head_dim"]
        num_kv_heads = config["num_key_value_heads"]
        full_head_dim = config["head_dim"]
        conv_dim = (config["linear_key_head_dim"] * config["linear_num_key_heads"] * 2) + (
            config["linear_value_head_dim"] * config["linear_num_value_heads"]
        )
        kernel = config["linear_conv_kernel_dim"]
        cache_len = int(config.get("attn_cache_len", 4096))
        dtype = params["embed_tokens"]["weight"].dtype
        state = {
            "linear_conv": jnp.zeros((n_linear, conv_dim, kernel - 1), dtype=dtype),
            "linear_recurrent": jnp.zeros(
                (n_linear, num_v_heads, head_k_dim, head_v_dim), dtype=dtype
            ),
            "full_k_cache": jnp.zeros((n_full, cache_len, num_kv_heads, full_head_dim), dtype=dtype),
            "full_v_cache": jnp.zeros((n_full, cache_len, num_kv_heads, full_head_dim), dtype=dtype),
            "position": jnp.array(0, dtype=jnp.int32),
        }
        preview_layers = tuple(config.get("preview_layers", ()))
        if preview_layers:
            state["preview_hidden"] = jnp.zeros(
                (len(preview_layers), config["hidden_size"]), dtype=jnp.float32
            )
        return state

    @classmethod
    def embed(cls, common_params, tokens):
        return common_params.params["embed_tokens"]["weight"][tokens.ravel()]

    @classmethod
    def outhead(cls, common_params, x):
        x = call_submodule(Qwen35RMSNorm, "norm", common_params, x, common_params.frozen_params["rms_norm_eps"])
        return x @ common_params.params["lm_head"]["weight"].T

    @classmethod
    def _rope_cos_sin(cls, common_params, position):
        cfg = common_params.frozen_params
        head_dim = cfg["head_dim"]
        rotary_dim = int(head_dim * cfg["partial_rotary_factor"])
        rotary_dim = rotary_dim - (rotary_dim % 2)
        inv_freq = 1.0 / (
            cfg["rope_theta"] ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / max(1, rotary_dim))
        )
        freqs = position.astype(jnp.float32) * inv_freq
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return jnp.cos(emb), jnp.sin(emb)

    @classmethod
    def forward_seq(cls, common_params, x, state, length, new_starts):
        cfg = common_params.frozen_params
        linear_indices = cfg["linear_layer_indices"]
        full_indices = cfg["full_layer_indices"]
        eps = cfg["rms_norm_eps"]

        layer_types = cfg["layer_types"]
        n_layers = len(layer_types)
        preview_layers = tuple(cfg.get("preview_layers", ()))
        preview_layer_slots = {
            layer: slot for slot, layer in enumerate(preview_layers)
        }

        linear_layer_to_idx = {layer: i for i, layer in enumerate(linear_indices)}
        full_layer_to_idx = {layer: i for i, layer in enumerate(full_indices)}

        def token_step(carry, token_pack):
            curr_state = carry
            token, restart = token_pack
            pos = curr_state["position"]
            x_t = token[None, :]

            def _reset_state(s):
                reset = {
                    "linear_conv": jnp.zeros_like(s["linear_conv"]),
                    "linear_recurrent": jnp.zeros_like(s["linear_recurrent"]),
                    "full_k_cache": jnp.zeros_like(s["full_k_cache"]),
                    "full_v_cache": jnp.zeros_like(s["full_v_cache"]),
                    "position": jnp.array(0, dtype=jnp.int32),
                }
                if preview_layers:
                    reset["preview_hidden"] = jnp.zeros_like(s["preview_hidden"])
                return s | reset

            curr_state = jax.lax.cond(restart, _reset_state, lambda s: s, curr_state)
            pos = curr_state["position"]

            rope_cos, rope_sin = cls._rope_cos_sin(common_params, pos)

            for layer in range(n_layers):
                block_params = jax.tree.map(lambda a: a[layer], common_params.params["blocks"])
                block_keys = jax.tree.map(lambda a: a[layer], common_params.es_tree_key["blocks"])
                block_frozen = {**cfg, "current_layer": layer}
                block_common = common_params._replace(
                    params=block_params,
                    es_tree_key=block_keys,
                    frozen_params=block_frozen,
                )

                residual = x_t
                x_t = call_submodule(Qwen35RMSNorm, "input_layernorm", block_common, x_t, eps)

                if layer_types[layer] == "linear_attention":
                    lidx = linear_layer_to_idx[layer]
                    lin_params = jax.tree.map(lambda a: a[lidx], common_params.params["blocks"]["linear_attn"])
                    lin_keys = jax.tree.map(lambda a: a[lidx], common_params.es_tree_key["blocks"]["linear_attn"])
                    lin_common = common_params._replace(
                        params=lin_params,
                        es_tree_key=lin_keys,
                        frozen_params={**cfg, "current_layer": layer},
                    )
                    out, conv_s, rec_s = Qwen35LinearAttention._forward(
                        lin_common,
                        x_t,
                        curr_state["linear_conv"][lidx],
                        curr_state["linear_recurrent"][lidx],
                    )
                    curr_state = curr_state | {
                        "linear_conv": curr_state["linear_conv"].at[lidx].set(conv_s),
                        "linear_recurrent": curr_state["linear_recurrent"].at[lidx].set(rec_s),
                    }
                    x_t = out
                else:
                    fidx = full_layer_to_idx[layer]
                    attn_params = jax.tree.map(lambda a: a[fidx], common_params.params["blocks"]["self_attn"])
                    attn_keys = jax.tree.map(lambda a: a[fidx], common_params.es_tree_key["blocks"]["self_attn"])
                    attn_common = common_params._replace(
                        params=attn_params,
                        es_tree_key=attn_keys,
                        frozen_params={**cfg, "current_layer": layer},
                    )
                    cache_len = curr_state["full_k_cache"].shape[1]
                    pos_clip = jnp.minimum(pos, cache_len - 1)
                    x_t, k_new, v_new = Qwen35SelfAttention._forward(
                        attn_common,
                        x_t,
                        rope_cos,
                        rope_sin,
                        curr_state["full_k_cache"][fidx],
                        curr_state["full_v_cache"][fidx],
                        pos_clip,
                    )
                    curr_state = curr_state | {
                        "full_k_cache": curr_state["full_k_cache"].at[fidx].set(k_new),
                        "full_v_cache": curr_state["full_v_cache"].at[fidx].set(v_new),
                    }

                x_t = residual + x_t
                residual = x_t
                x_t = call_submodule(Qwen35RMSNorm, "post_attention_layernorm", block_common, x_t, eps)
                x_t = call_submodule(Qwen35MLP, "mlp", block_common, x_t)
                x_t = residual + x_t

                if layer in preview_layer_slots:
                    slot = preview_layer_slots[layer]
                    old_hidden = curr_state["preview_hidden"][slot]
                    captured_hidden = jnp.where(
                        pos == length - 1, x_t[0].astype(jnp.float32), old_hidden
                    )
                    curr_state = curr_state | {
                        "preview_hidden": curr_state["preview_hidden"]
                        .at[slot]
                        .set(captured_hidden)
                    }

            curr_state = curr_state | {"position": curr_state["position"] + 1}
            return curr_state, x_t[0]

        final_state, y = jax.lax.scan(token_step, state, (x, new_starts))
        return y, final_state
