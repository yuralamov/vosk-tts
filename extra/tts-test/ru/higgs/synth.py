#!/usr/bin/env python3
"""
Standalone Higgs Audio v3 TTS synthesizer (no ComfyUI, no SGLang).

This is a self-contained port of the native inference path from
https://github.com/Saganaki22/Higgs_v3-TTS-ComfyUI (loader.py / native.py),
with all ComfyUI memory-management plumbing removed. It loads
`bosonai/higgs-audio-v3-tts-4b` directly with plain transformers and runs the
custom audio-codec + delay-pattern decoding loop in-process.

Model is a Qwen3 text backbone that, instead of (or alongside) text tokens,
autoregressively emits N parallel codebooks of audio codes. Those codes are
written with a "delay pattern", reversed, and decoded to a 24 kHz waveform by
the bundled HiggsAudioV2 tokenizer (codec), whose weights live inside the same
model.safetensors under a `tied.embedding.modality_embeddings.0.model.` prefix.

Requirements (the native path targets Transformers 5.3.0-5.5.0; 5.5.0 best):
    pip install "transformers>=5.3,<5.6" torch torchaudio safetensors \
        tokenizers huggingface_hub accelerate

Usage:
    python higgs_v3_tts.py "Hello, how are you today?"
    python higgs_v3_tts.py "Have a nice day." -o hello.wav
    # zero-shot voice cloning (a correct transcript improves cloning a lot):
    python higgs_v3_tts.py "Hey there, nice to meet you." \
        --ref-audio ref.wav --ref-text "Hey, Adam here. Let's make something real."

Inline control tokens work inside the text, e.g.:
    "<|emotion:amusement|><|prosody:expressive_high|>Wait, that was hilarious. <|sfx:laughter|>Haha."
Always pair every <|sfx:...|> with its onomatopoeia (e.g. <|sfx:laughter|>Haha).

First run downloads ~9.3 GB of weights from HuggingFace into ./higgs-v3-model
(override with --model-dir). bf16 inference needs ~11 GB VRAM; on CPU it runs in
fp32 (slow). Disk for the model only; the codec config is embedded below.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from timeit import default_timer as timer
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import torchaudio
from safetensors import safe_open
from tokenizers import Tokenizer
from torch import nn
from transformers import (
    HiggsAudioV2TokenizerConfig,
    HiggsAudioV2TokenizerModel,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

logging.basicConfig(level=logging.INFO, format="[higgs-v3] %(message)s")
logger = logging.getLogger("higgs-v3")

# --- Constants (from native.py) ----------------------------------------------
REPO_ID = "bosonai/higgs-audio-v3-tts-4b"
BOC_ID = 1024            # begin-of-codebook filler
EOC_ID = 1025            # end-of-codebook / stop marker
STOP_CODE = -1
AUDIO_PLACEHOLDER_ID = -100
SAMPLE_RATE = 24_000
# Codec weights are stored inside model.safetensors under this prefix.
CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."

# The audio-codec (tokenizer) config is a repo asset of the ComfyUI nodepack,
# not part of the HF model repo, so it is embedded here verbatim.
CODEC_CONFIG: dict[str, Any] = {
    "acoustic_model_config": {
        "codebook_dim": 8, "codebook_loss_weight": 1.0, "codebook_size": 1024,
        "commitment_loss_weight": 0.25, "decoder_hidden_size": 1024,
        "downsampling_ratios": [8, 5, 4, 2, 3], "encoder_hidden_size": 64,
        "hidden_size": 256, "hop_length": 960, "model_type": "dac", "n_codebooks": 9,
        "quantizer_dropout": 0, "sampling_rate": 16000,
        "upsampling_ratios": [8, 5, 4, 2, 3],
    },
    "block_dilations": [1, 1], "channel_ratios": [1, 1], "codebook_dim": 64,
    "codebook_size": 1024, "downsample_factor": 320, "initializer_range": 0.02,
    "kernel_size": 3, "model_type": "higgs_audio_v2_tokenizer", "sample_rate": 24000,
    "semantic_model_config": {
        "activation_dropout": 0.1, "apply_spec_augment": True, "attention_dropout": 0.1,
        "bos_token_id": 1, "classifier_proj_size": 256, "conv_bias": False,
        "conv_dim": [512, 512, 512, 512, 512, 512, 512],
        "conv_kernel": [10, 3, 3, 3, 3, 2, 2], "conv_pos_batch_norm": False,
        "conv_stride": [5, 2, 2, 2, 2, 2, 2], "ctc_loss_reduction": "sum",
        "ctc_zero_infinity": False, "do_stable_layer_norm": False, "eos_token_id": 2,
        "feat_extract_activation": "gelu", "feat_extract_norm": "group",
        "feat_proj_dropout": 0.0, "feat_proj_layer_norm": True, "final_dropout": 0.1,
        "hidden_act": "gelu", "hidden_dropout": 0.1, "hidden_size": 768,
        "initializer_range": 0.02, "intermediate_size": 3072, "layer_norm_eps": 1e-05,
        "layerdrop": 0.1, "mask_feature_length": 10, "mask_feature_min_masks": 0,
        "mask_feature_prob": 0.0, "mask_time_length": 10, "mask_time_min_masks": 2,
        "mask_time_prob": 0.0, "model_type": "hubert", "num_attention_heads": 12,
        "num_conv_pos_embedding_groups": 16, "num_conv_pos_embeddings": 128,
        "num_feat_extract_layers": 7, "num_hidden_layers": 12, "pad_token_id": 0,
        "use_weighted_layer_sum": False, "vocab_size": 32,
    },
    "semantic_sample_rate": 16000, "strides": [1, 1],
    "target_bandwidths": [0.5, 1, 1.5, 2], "unit_kernel_size": 3,
}


# --- Fused multi-codebook embedding / head -----------------------------------
class HiggsFusedMultiTextEmbedding(nn.Module):
    """Sum-embeds N codebooks at once via a single offset-indexed weight table."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = int(num_codebooks)
        self.vocab_size = int(vocab_size)

    def forward(self, codes_l_n: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(
            self.num_codebooks, device=codes_l_n.device, dtype=codes_l_n.dtype
        ) * self.vocab_size
        return F.embedding(codes_l_n + offsets, self.weight).sum(dim=-2)


class HiggsFusedMultiTextHead(nn.Module):
    """Projects hidden states to per-codebook logits [L, N, V]."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = int(num_codebooks)
        self.vocab_size = int(vocab_size)

    def generate(self, hidden_l_d: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_l_d, self.weight)
        return logits.reshape(hidden_l_d.shape[0], self.num_codebooks, self.vocab_size)


# --- Sampling + delay-pattern state machine ----------------------------------
@dataclass
class HiggsSamplerState:
    num_codebooks: int
    delay_count: int = 0
    eoc_countdown: int | None = None
    generation_done: bool = False
    last_codes: torch.Tensor | None = None


def _sample_independent(logits_n_v, *, temperature, top_p, top_k):
    if temperature <= 1e-5:
        return logits_n_v.argmax(dim=-1)
    logits = logits_n_v / float(temperature)
    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        kth = logits.topk(k, dim=-1).values[:, -1:]
        logits = torch.where(logits < kth, float("-inf"), logits)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum_probs > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter = torch.zeros_like(remove)
        scatter.scatter_(-1, sorted_indices, remove)
        logits = torch.where(scatter, float("-inf"), logits)
    probs = logits.softmax(dim=-1)
    return probs.multinomial(num_samples=1).squeeze(-1)


def sampler_step(logits_n_v, state: HiggsSamplerState, *, temperature, top_p, top_k):
    """Sample one frame of N codebook codes, advancing the delay/EOC state."""
    n = state.num_codebooks
    if state.generation_done:
        return torch.full((n,), STOP_CODE, dtype=torch.long, device=logits_n_v.device)

    codes_n = _sample_independent(
        logits_n_v, temperature=temperature, top_p=top_p, top_k=top_k
    ).to(torch.long)

    if state.delay_count < n:
        next_cb = state.delay_count + 1
        if next_cb < n:
            codes_n[next_cb:] = BOC_ID
        state.delay_count += 1
    elif state.eoc_countdown is not None:
        state.eoc_countdown -= 1
        if state.eoc_countdown <= 0:
            state.generation_done = True
    elif int(codes_n[0].item()) == EOC_ID:
        state.eoc_countdown = n - 2 if n > 2 else 0
        if n <= 2:
            state.generation_done = True

    if not state.generation_done:
        state.last_codes = codes_n.clone()
    return codes_n


def apply_delay_pattern(codes_t_n: torch.Tensor) -> torch.Tensor:
    if codes_t_n.ndim != 2:
        raise ValueError(f"codes must be [T, N], got {tuple(codes_t_n.shape)}")
    t, n = codes_t_n.shape
    out = torch.full((t + n - 1, n), EOC_ID, device=codes_t_n.device, dtype=codes_t_n.dtype)
    t_idx = torch.arange(t + n - 1, device=codes_t_n.device)
    for c in range(n):
        out[t_idx < c, c] = BOC_ID
        out[c : c + t, c] = codes_t_n[:, c]
    return out


def reverse_delay_pattern(delayed_l_n: torch.Tensor) -> torch.Tensor:
    if delayed_l_n.ndim != 2:
        raise ValueError(f"delayed codes must be [L, N], got {tuple(delayed_l_n.shape)}")
    l, n = delayed_l_n.shape
    t = l - (n - 1)
    if t <= 0:
        raise ValueError(f"Need at least {n} delayed rows, got {l}.")
    out = torch.empty((t, n), device=delayed_l_n.device, dtype=delayed_l_n.dtype)
    for c in range(n):
        out[:, c] = delayed_l_n[c : c + t, c]
    return out


# --- Tokenizer adapter -------------------------------------------------------
class HiggsTokenizerAdapter:
    """Builds the <|tts|> ... <|text|> ... <|audio|> prompt token sequence."""

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        vocab = dict(tokenizer.get_added_vocab())
        missing = [t for t in ("<|tts|>", "<|ref_audio|>", "<|text|>", "<|audio|>") if t not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing Higgs TTS specials: {missing}")
        self.tts_id = vocab["<|tts|>"]
        self.ref_audio_id = vocab["<|ref_audio|>"]
        self.text_id = vocab["<|text|>"]
        self.audio_id = vocab["<|audio|>"]
        self.ref_text_id = vocab.get("<|ref_text|>")

    def build_prompt(self, prompt_text, *, num_ref_tokens=0, reference_text=None) -> list[int]:
        ids = [self.tts_id]
        if reference_text and num_ref_tokens > 0 and self.ref_text_id is not None:
            ids.append(self.ref_text_id)
            ids.extend(self._tok.encode(reference_text, add_special_tokens=False))
        if num_ref_tokens > 0:
            ids.append(self.ref_audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * int(num_ref_tokens))
        ids.append(self.text_id)
        ids.extend(self._tok.encode(prompt_text, add_special_tokens=False))
        ids.append(self.audio_id)
        return ids


def load_tokenizer(model_dir: Path) -> HiggsTokenizerAdapter:
    raw = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return HiggsTokenizerAdapter(PreTrainedTokenizerFast(tokenizer_object=raw))


# --- Native TTS model (Qwen3 backbone + audio codebook head) -----------------
class HiggsNativeTTS(nn.Module):
    def __init__(self, config: dict[str, Any], torch_dtype: torch.dtype, attn_implementation: str | None):
        super().__init__()
        text_config = Qwen3Config(**dict(config["text_config"]))
        if attn_implementation is not None:
            text_config._attn_implementation = attn_implementation
        self.backbone = Qwen3ForCausalLM(text_config)
        self.backbone.to(dtype=torch_dtype)

        enc_cfg = dict(config.get("audio_encoder_config") or {})
        if enc_cfg.get("encoder_type", "discrete") != "discrete":
            raise NotImplementedError("Only the Higgs v3 discrete TTS path is supported.")
        self.num_codebooks = int(enc_cfg.get("num_codebooks", 8))
        self.codebook_vocab_size = int(enc_cfg.get("vocab_size", 1026))
        hidden_size = int(enc_cfg.get("out_dim", text_config.hidden_size))
        self.modality_embedding = HiggsFusedMultiTextEmbedding(
            self.num_codebooks, self.codebook_vocab_size, hidden_size
        ).to(dtype=torch_dtype)
        self.modality_head = HiggsFusedMultiTextHead(
            self.num_codebooks, self.codebook_vocab_size, hidden_size
        ).to(dtype=torch_dtype)
        self.tie_modality = bool(enc_cfg.get("tie_word_embeddings", True))
        self.retie_weights()

    def retie_weights(self) -> None:
        if self.tie_modality:
            self.modality_head.weight = self.modality_embedding.weight
        try:
            self.backbone.tie_weights()
        except Exception:
            pass

    def _prompt_embeds(self, prompt_ids, reference_codes_delayed, device) -> torch.Tensor:
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        mask = ids == AUDIO_PLACEHOLDER_ID
        safe_ids = ids.clamp_min(0).view(1, -1)
        embeds = self.backbone.model.embed_tokens(safe_ids)
        if mask.any():
            if reference_codes_delayed is None:
                raise ValueError("Prompt has audio placeholders but no reference codes.")
            ref_codes = reference_codes_delayed.to(device=device, dtype=torch.long)
            if int(mask.sum().item()) != ref_codes.shape[0]:
                raise ValueError("Reference placeholder count != delayed codes length.")
            embeds[0, mask] = self.modality_embedding(ref_codes).to(embeds.dtype)
        return embeds

    @torch.no_grad()
    def generate_codes(self, prompt_ids, reference_codes_delayed, *,
                       max_new_tokens, temperature, top_p, top_k) -> torch.Tensor:
        device = next(self.parameters()).device
        prompt_embeds = self._prompt_embeds(prompt_ids, reference_codes_delayed, device)
        out = self.backbone.model(inputs_embeds=prompt_embeds, use_cache=True)
        past = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]
        state = HiggsSamplerState(num_codebooks=self.num_codebooks)
        rows: list[torch.Tensor] = []

        for i in range(int(max_new_tokens)):
            logits = self.modality_head.generate(hidden)[0].to(torch.float32)
            codes = sampler_step(logits, state, temperature=float(temperature), top_p=top_p, top_k=top_k)
            if int(codes[0].item()) != STOP_CODE:
                rows.append(codes.detach().to("cpu", torch.long))
            if i == 0 or (i + 1) % 128 == 0 or state.generation_done:
                logger.info("audio tokens %d/%d", i + 1, int(max_new_tokens))
            if state.generation_done or state.last_codes is None:
                break
            next_embed = self.modality_embedding(state.last_codes.view(1, -1)).view(1, 1, -1)
            out = self.backbone.model(
                inputs_embeds=next_embed.to(device=device, dtype=prompt_embeds.dtype),
                past_key_values=past, use_cache=True,
            )
            past = out.past_key_values
            hidden = out.last_hidden_state[:, -1, :]

        if len(rows) < self.num_codebooks:
            raise RuntimeError(f"Generated too few audio token rows ({len(rows)}).")
        if not state.generation_done and len(rows) >= int(max_new_tokens):
            logger.warning("Hit max_new_tokens before a stop token; raise --max-new-tokens for longer text.")
        return torch.stack(rows, dim=0)


# --- Audio codec (decode codes -> waveform; encode reference for cloning) -----
def _codec_config() -> HiggsAudioV2TokenizerConfig:
    return HiggsAudioV2TokenizerConfig(**CODEC_CONFIG)


def _codec_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards: dict[str, list[str]] = {}
        for full_name, shard in weight_map.items():
            if full_name.startswith(CODEC_PREFIX):
                shards.setdefault(shard, []).append(full_name)
    else:
        shards = {"model.safetensors": []}

    state: dict[str, torch.Tensor] = {}
    for shard, names in shards.items():
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            keys = names or [k for k in f.keys() if k.startswith(CODEC_PREFIX)]
            for full_name in keys:
                state[full_name[len(CODEC_PREFIX):]] = f.get_tensor(full_name)
    if not state:
        raise FileNotFoundError(f"No bundled codec weights found in {model_dir}.")
    return state


class HiggsAudioCodec:
    def __init__(self, model: HiggsAudioV2TokenizerModel, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.dtype = next(model.parameters()).dtype

    @classmethod
    def from_pretrained(cls, model_dir: Path, *, device, dtype) -> "HiggsAudioCodec":
        model = HiggsAudioV2TokenizerModel(_codec_config()).to(dtype=dtype).eval()
        state = _codec_state_dict(model_dir)
        missing, _ = model.load_state_dict(state, strict=False)
        if len(missing) > len(state) // 2:
            raise RuntimeError(f"Codec load too sparse: {len(missing)} missing / {len(state)} loaded.")
        model = model.to(device=device)
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, device)

    @torch.no_grad()
    def encode_reference(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav = waveform.detach().float().cpu()
        if wav.ndim == 1:
            wav = wav.view(1, 1, -1)
        elif wav.ndim == 2:
            wav = wav[:1].unsqueeze(0)
        elif wav.ndim == 3:
            wav = wav[:, :1, :]
        else:
            raise ValueError(f"Unsupported reference audio shape: {tuple(wav.shape)}")
        if int(sample_rate) != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, int(sample_rate), SAMPLE_RATE)
        if wav.shape[-1] < SAMPLE_RATE:
            wav = F.pad(wav, (0, SAMPLE_RATE - wav.shape[-1]))
        wav = wav.to(device=self.device, dtype=self.dtype)
        codes_b_n_t = self.model.encode(wav).audio_codes
        return codes_b_n_t.squeeze(0).transpose(0, 1).to(torch.long).cpu()

    @torch.no_grad()
    def decode(self, codes_t_n: torch.Tensor) -> torch.Tensor:
        codes_b_n_t = codes_t_n.transpose(0, 1).unsqueeze(0).to(device=self.device, dtype=torch.long)
        return self.model.decode(codes_b_n_t).audio_values.squeeze(0).squeeze(0).detach().float().cpu()


# --- Weight loading ----------------------------------------------------------
def map_higgs_weight_name(name: str) -> str | None:
    if name.startswith(CODEC_PREFIX):
        return None
    prefix_map = {
        "tied.embedding.text_embedding.": "backbone.model.embed_tokens.",
        "body.layers.": "backbone.model.layers.",
        "body.norm.": "backbone.model.norm.",
        "tied.head.text_head.": "backbone.lm_head.",
        "tied.embedding.modality_embeddings.0.embedding.": "modality_embedding.",
        "tied.head.modality_heads.0.": "modality_head.",
    }
    for source, target in prefix_map.items():
        if name.startswith(source):
            return target + name[len(source):]
    return name


def iter_safetensor_items(model_dir: Path) -> Iterable[tuple[str, torch.Tensor]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]
    for shard in shards:
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def _set_tensor(module: nn.Module, name: str, tensor: torch.Tensor, device, dtype) -> None:
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    try:
        from accelerate.utils.modeling import set_module_tensor_to_device

        set_module_tensor_to_device(module, name, device=device, value=tensor.contiguous())
        return
    except Exception:
        pass
    target = dict(module.named_parameters(remove_duplicate=False)).get(name)
    if target is None:
        target = dict(module.named_buffers(remove_duplicate=False)).get(name)
    if target is None:
        raise KeyError(name)
    if target.shape != tensor.shape:
        raise ValueError(f"Shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}")
    target.data = tensor.to(device=device).contiguous()


def build_native_model(config, dtype, attention_impl) -> HiggsNativeTTS:
    try:
        from accelerate import init_empty_weights

        with init_empty_weights():
            return HiggsNativeTTS(config, dtype, attention_impl)
    except Exception:
        return HiggsNativeTTS(config, dtype, attention_impl)


def load_native_weights(model: HiggsNativeTTS, model_dir: Path, device, dtype) -> None:
    param_names = set(dict(model.named_parameters(remove_duplicate=False)))
    loaded: set[str] = set()
    for source_name, tensor in iter_safetensor_items(model_dir):
        mapped = map_higgs_weight_name(source_name)
        if mapped is None:
            continue
        if mapped not in param_names:
            continue
        _set_tensor(model, mapped, tensor, device, dtype)
        loaded.add(mapped)
    model.retie_weights()
    model.to(device=device)
    model.eval()
    missing = [n for n in param_names
               if "lm_head.weight" not in n and "modality_head.weight" not in n and n not in loaded]
    if missing:
        raise RuntimeError(f"Native load missed {len(missing)} param(s), first: {missing[:8]}")
    logger.info("Loaded %d native weight tensors.", len(loaded))


# --- Audio helpers -----------------------------------------------------------
def trim_silence_edges(audio: torch.Tensor, sample_rate: int, threshold_db: float = -42.0) -> torch.Tensor:
    if audio.numel() == 0:
        return audio
    threshold = 10 ** (threshold_db / 20.0)
    frame = max(1, int(sample_rate * 0.01))
    padded = F.pad(audio.abs(), (0, (frame - audio.numel() % frame) % frame))
    rms = padded.view(-1, frame).pow(2).mean(dim=1).sqrt()
    active = torch.nonzero(rms > threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        return audio
    start = max(0, int(active[0].item()) * frame)
    end = min(audio.numel(), (int(active[-1].item()) + 1) * frame)
    pad = int(sample_rate * 0.05)
    return audio[max(0, start - pad): min(audio.numel(), end + pad)].contiguous()


def load_audio_path(path: str) -> tuple[torch.Tensor, int]:
    audio_path = Path(path).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")
    wav, sr = torchaudio.load(str(audio_path))
    wav = wav.mean(dim=0) if (wav.ndim == 2 and wav.shape[0] > 1) else wav.squeeze(0)
    return wav.detach().float().cpu().contiguous(), int(sr)


# --- Bundle + top-level synthesis --------------------------------------------
@dataclass
class HiggsBundle:
    model: HiggsNativeTTS
    codec: HiggsAudioCodec
    tokenizer: HiggsTokenizerAdapter
    device: torch.device
    attn: str


def download_model(model_dir: Path) -> Path:
    """Fetch the HF weights + configs into model_dir (skips files already present)."""
    if (model_dir / "model.safetensors").is_file() and (model_dir / "config.json").is_file():
        return model_dir
    from huggingface_hub import snapshot_download

    logger.info("Downloading %s into %s (~9.3 GB on first run)...", REPO_ID, model_dir)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(model_dir),
        allow_patterns=[
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors", "model.safetensors.index.json",
        ],
    )
    return model_dir


def load_bundle(model_dir: Path, device_name: str, dtype_name: str, attn: str) -> HiggsBundle:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda is unavailable.")

    if dtype_name == "bf16" or (dtype_name == "auto" and device.type == "cuda"
                                and torch.cuda.is_bf16_supported()):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    logger.info("Loading on %s with dtype=%s, attention=%s", device, dtype, attn)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = build_native_model(config, dtype, attn)
    load_native_weights(model, model_dir, device, dtype)
    codec = HiggsAudioCodec.from_pretrained(model_dir, device=device, dtype=dtype)
    tokenizer = load_tokenizer(model_dir)
    return HiggsBundle(model=model, codec=codec, tokenizer=tokenizer, device=device, attn=attn)


def synthesize(bundle: HiggsBundle, text: str, *, reference_audio=None, reference_text="",
               max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=50,
               seed=None, trim_reference=True) -> torch.Tensor:
    """Return a 1-D 24 kHz waveform tensor for `text`."""
    if not text.strip():
        raise ValueError("Text cannot be empty.")
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))

    ref_delayed = None
    if reference_audio:
        wav, sr = load_audio_path(reference_audio)
        if trim_reference:
            wav = trim_silence_edges(wav, sr)
        raw_codes = bundle.codec.encode_reference(wav, sr)
        ref_delayed = apply_delay_pattern(raw_codes)

    prompt_ids = bundle.tokenizer.build_prompt(
        text.strip(),
        num_ref_tokens=0 if ref_delayed is None else int(ref_delayed.shape[0]),
        reference_text=reference_text.strip() or None,
    )
    with torch.inference_mode():
        delayed = bundle.model.generate_codes(
            prompt_ids, ref_delayed,
            max_new_tokens=int(max_new_tokens), temperature=float(temperature),
            top_p=None if top_p <= 0 or top_p >= 1 else float(top_p),
            top_k=None if top_k <= 0 else int(top_k),
        )
        raw = reverse_delay_pattern(delayed)
        audio = bundle.codec.decode(raw)
    if not torch.isfinite(audio).all():
        raise RuntimeError("Generated non-finite audio samples.")
    return audio.clamp(-1.0, 1.0)



bundle = load_bundle(Path("higgs-audio-v3-tts-4b"), "cuda", "bf16", "sdpa")

spkmap = {}
for i, line in enumerate(open("eval-speakers-text/metadata-phones-ids.csv.test-ref")):
    items = line.strip().split("|")
    spk = items[0].split("/")[0]
    spkmap[spk] = (items[0].replace("/", "_"), items[1])

def main_synth(out_dir=None):

    os.makedirs(out_dir, exist_ok=True)

    start = timer()
    total_len = 0
    for i, line in enumerate(open("eval.csv", encoding='utf-8')):
        items = line.strip().split("|")
        fitems = items[0].split("/")
        uid = fitems[2] + "_" + fitems[-1][:-4]
        spk = items[0].split("/")[2]
        text = items[1].replace("+", "")

        reffn, reftext = spkmap[spk]
        reftext = reftext.replace("+", "")
        reffn = "eval-speakers-text/wav/" + reffn

        audio = synthesize(
            bundle, text,
            reference_audio=reffn, reference_text=reftext,
            max_new_tokens=1024, temperature=0.8,
            top_p=0.95, top_k=50, seed=None,
        )
        out_wav = audio.view(1, -1)
        total_len += out_wav.size(1)

        torchaudio.save(out_dir + "/" + uid + ".wav", out_wav, SAMPLE_RATE)

    end = timer()

    audio_duration_sec = float(total_len) / SAMPLE_RATE
    infer_sec = end - start
    real_time_factor = (infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0)
    print(f"Real-time factor: {real_time_factor:.4f} (infer={infer_sec:.2f} sec, audio={audio_duration_sec:.2f} sec)")
    
main_synth(out_dir = 'out')
