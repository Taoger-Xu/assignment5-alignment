import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer


def save_training_checkpoint(path, model, tokenizer, optimizer, step, trainer_rng,
                             trainer_config, sampling_config, run_args=None):
    """Atomically publish model, optimizer and RNG state at a rollout boundary."""
    import json
    import os
    import random
    import tempfile
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=path.parent))
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(temporary, max_shard_size="100GB")
    else:
        torch.save(model.state_dict(), temporary / "pytorch_model.bin")
    if tokenizer is not None:
        tokenizer.save_pretrained(temporary)
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "trainer_rng": trainer_rng.getstate(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, temporary / "training_state.pt")
    (temporary / "checkpoint_config.json").write_text(json.dumps({
        "step": step, "trainer_config": trainer_config,
        "sampling_config": sampling_config, "run_args": run_args or {},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    os.rename(temporary, path)
    return path


def load_training_checkpoint(path, model, optimizer, trainer_rng):
    """Restore a locally created checkpoint; return completed rollout step."""
    import random
    from pathlib import Path

    path = Path(path)
    if (path / "model.safetensors").exists():
        from safetensors.torch import load_model
        load_model(model, str(path / "model.safetensors"), device="cpu")
    else:
        model.load_state_dict(torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True))
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    trainer_rng.setstate(state["trainer_rng"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"])
