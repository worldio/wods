"""Script to add Qwen text embeddings to wods datasets.

Reads the caption from each wods item's metadata, encodes it with the
Qwen2.5-VL text encoder (``Qwen2_5_VLTextModel``, hidden_size=3584)
from the HunyuanVideo 1.5 diffusers repo, and writes the resulting embedding
as ``text_embed`` to every shard of that item.

By default, each complete key uses its own caption. ``--group-final-components N``
opts into sharing the first key's caption (in selected/listed order) among keys
whose paths match after dropping the final N components: 0 keeps full keys,
1 groups siblings (the previous behavior), and 2 groups by the grandparent path.
N must be nonnegative and leave at least one component in every selected key;
otherwise the run fails before model loading or writing. Whole groups are
assigned to distributed ranks so a group always uses the same caption.

The encoding exactly matches ``HunyuanVideo15Pipeline._get_mllm_prompt_embeds``:
  - Wraps the caption in a chat template with the HunyuanVideo system message.
  - Uses ``hidden_states[-3]`` (skipping the last 2 layers).
  - Crops the first 108 tokens (system prefix) from the output.

The default model is the ``text_encoder`` subfolder of
``hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v``.
Any HuggingFace text model can be substituted via ``--model``.

Usage:
    uv run python scripts/add_text_embed_qwen.py \\
        --input /path/to/input/wods --output /path/to/output/wods

    # Process a specific key
    uv run python scripts/add_text_embed_qwen.py \\
        --input /data/wods --output /data/wods_out --keys episode_00000

    # Opt in to sharing a caption across sibling keys
    PYTHONUNBUFFERED=1 uv run python scripts/add_text_embed_qwen.py \\
        --input /data/wods --output /data/wods_out --group-final-components 1

    # Use a different model (no subfolder)
    uv run python scripts/add_text_embed_qwen.py \\
        --input /data/wods --output /data/wods_out \\
        --model Qwen/Qwen2.5-0.5B --model-subfolder "" --tokenizer-subfolder ""

    # Choose storage dtype
    uv run python scripts/add_text_embed_qwen.py \\
        --input /data/wods --output /data/wods_out --storage-dtype float16

    # Distributed processing
    RANK=0 WORLD_SIZE=4 uv run python scripts/add_text_embed_qwen.py \\
        --input /data/wods --output /data/wods_out
"""

import argparse
import concurrent.futures
import os

import torch
from rich import print
from tqdm import tqdm

from wods.v2 import (
    list_wods_keys,
    list_wods_shards,
    read_wods_metadata,
    read_wods_shard,
    wods_key_exists,
    write_wods_key_done,
    write_wods_metadata,
    write_wods_shard,
    write_wods_shard_done,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add Qwen text embeddings to wods datasets",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input wods dataset path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output wods dataset path (can be the same as input to update in-place)",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Specific keys to process in one model-loading run",
    )
    parser.add_argument(
        "--group-final-components",
        type=int,
        default=0,
        metavar="N",
        help="Number of final '/'-separated key components to drop when grouping: "
        "0 keeps full keys independent (default), 1 groups siblings (previous behavior). "
        "Each group uses its first key's caption. Must be nonnegative and leave at least "
        "one component in every selected key.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
        help="HuggingFace repo ID for the text encoder "
        "(default: hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v)",
    )
    parser.add_argument(
        "--model-subfolder",
        type=str,
        default="text_encoder",
        help="Subfolder within the repo that contains the model weights "
        "(default: text_encoder). Set to empty string for top-level models.",
    )
    parser.add_argument(
        "--tokenizer-subfolder",
        type=str,
        default="tokenizer",
        help="Subfolder within the repo that contains the tokenizer "
        "(default: tokenizer). Set to empty string for top-level models.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model compute dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--storage-dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for stored embeddings (default: float32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (e.g. cuda:0, cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Maximum token length for user content (default: 1000, matching HunyuanVideo 1.5)",
    )
    parser.add_argument(
        "--num-hidden-layers-to-skip",
        type=int,
        default=2,
        help="Number of final hidden layers to skip (default: 2, matching HunyuanVideo 1.5)",
    )
    parser.add_argument(
        "--crop-start",
        type=int,
        default=108,
        help="Number of system-prefix tokens to crop from the output (default: 108, matching HunyuanVideo 1.5)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="Number of parallel I/O workers for writing shards (default: 8)",
    )
    args = parser.parse_args()
    if args.group_final_components < 0:
        parser.error("--group-final-components must be nonnegative")
    return args


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def setup_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1
    return rank, world_size


# ---------------------------------------------------------------------------
# Constants — must match HunyuanVideo15Pipeline defaults
# ---------------------------------------------------------------------------

# fmt: off
# NOTE: must exactly match the default in HunyuanVideo15Pipeline._get_mllm_prompt_embeds
# (the backslash-continuation + 8-space indent produces 9 spaces between items).
SYSTEM_MESSAGE = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."
# fmt: on


def _format_text_input(prompts: list[str], system_message: str) -> list[list[dict[str, str]]]:
    """Wrap each prompt in the chat template expected by HunyuanVideo 1.5."""
    return [
        [{"role": "system", "content": system_message}, {"role": "user", "content": p if p else " "}] for p in prompts
    ]


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_texts(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 1000,
    num_hidden_layers_to_skip: int = 2,
    crop_start: int = 108,
) -> list[torch.Tensor]:
    """Encode a batch of strings → list of (L_i, D) tensors on CPU.

    Mirrors ``HunyuanVideo15Pipeline._get_mllm_prompt_embeds`` exactly:
      1. Wrap each text in a chat template with *SYSTEM_MESSAGE*.
      2. Tokenise with ``apply_chat_template``.
      3. Extract ``hidden_states[-(num_hidden_layers_to_skip + 1)]``.
      4. Crop the first *crop_start* tokens (system prefix).
      5. Strip padding from each sequence.

    Returns one ``(L_i, D)`` tensor per input text.
    """
    conversations = _format_text_input(texts, SYSTEM_MESSAGE)

    text_inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding="max_length",
        max_length=max_length + crop_start,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device=device)
    attention_mask = text_inputs.attention_mask.to(device=device)

    outputs = model(
        input_ids=text_input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden = outputs.hidden_states[-(num_hidden_layers_to_skip + 1)]  # (B, L, D)

    # Crop system-prefix tokens
    if crop_start > 0:
        hidden = hidden[:, crop_start:]
        attention_mask = attention_mask[:, crop_start:]

    results: list[torch.Tensor] = []
    for i in range(hidden.size(0)):
        seq_len = int(attention_mask[i].sum())
        results.append(hidden[i, :seq_len].cpu())  # (L_i, D)

    return results


# ---------------------------------------------------------------------------
# Per-key processing
# ---------------------------------------------------------------------------


def write_embed_for_key(
    key: str,
    embed: torch.Tensor | None,
    args,
):
    """Copy existing shard data and add text_embed to every shard of a key."""
    try:
        metadata = read_wods_metadata(args.input, key)
        write_wods_metadata(
            args.output,
            key,
            **{k: v for k, v in metadata.items() if k != "__key__"},
        )

        shards = list(list_wods_shards(args.input, key))
        for shard in shards:
            shard_data = read_wods_shard(
                args.input,
                key,
                shard,
                include_video=False,
                include_video_raw=True,
                include_depth=False,
                include_depth_raw=True,
                include_text_embed=False,
            )
            if embed is not None:
                shard_data["text_embed"] = embed
            write_wods_shard(args.output, key, shard, **shard_data)
            write_wods_shard_done(args.output, key, shard)

        write_wods_key_done(args.output, key)
    except Exception as e:
        print(f"  Error writing key {key}: {e}")
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _write_key(key: str, embed: torch.Tensor | None, args) -> str:
    write_embed_for_key(key, embed, args)
    return "written"


def _selected_keys(args) -> list[str] | None:
    if args.keys is not None:
        if len(set(args.keys)) != len(args.keys):
            raise ValueError("--keys must not contain duplicates")
        return args.keys
    return None


def _process_episodes(episode_to_keys, get_embed, args, pbar):
    """Prepare captions sequentially while a bounded pool processes keys."""
    counts = {"written": 0, "skipped": 0, "failed": 0, "no_caption": 0}
    max_inflight = max(args.num_workers * 4, 1)
    pending = {}

    def collect(done):
        for future in done:
            key = pending.pop(future)
            try:
                counts[future.result()] += 1
            except Exception as e:
                counts["failed"] += 1
                print(f"  Error writing key {key}: {e}")
            pbar.update(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for _episode, keys in episode_to_keys.items():
            remaining = []
            for key in keys:
                try:
                    exists = wods_key_exists(args.output, key)
                except Exception as e:
                    counts["failed"] += 1
                    print(f"  Error checking key {key}: {e}")
                    pbar.update(1)
                    continue
                if exists:
                    counts["skipped"] += 1
                    pbar.update(1)
                else:
                    remaining.append(key)
            if not remaining:
                continue
            try:
                metadata = read_wods_metadata(args.input, keys[0])
                caption = metadata.get("caption", None)
            except Exception as e:
                print(f"  Error reading metadata for {keys[0]}: {e}")
                counts["failed"] += len(remaining)
                pbar.update(len(remaining))
                continue

            if caption is None or caption == "":
                counts["no_caption"] += len(remaining)
                embed = None
            else:
                embed = get_embed(caption)

            for key in remaining:
                while len(pending) >= max_inflight:
                    done, _ = concurrent.futures.wait(
                        pending,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    collect(done)
                future = executor.submit(_write_key, key, embed, args)
                pending[future] = key

        for future in concurrent.futures.as_completed(list(pending)):
            collect([future])

    return counts


def main():
    args = parse_args()
    rank, world_size = setup_distributed()

    # Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Rank {rank}/{world_size}, Device: {device}")

    # Dtype map
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]
    storage_dtype = dtype_map[args.storage_dtype]

    # Collect keys to process
    selected_keys = _selected_keys(args)
    if selected_keys is not None:
        all_keys = selected_keys
        print(f"Processing {len(selected_keys)} specific key(s)")
    else:
        all_keys = list(list_wods_keys(args.input))

    # Group before rank assignment so each group uses one first-key caption.
    from collections import defaultdict

    episode_to_keys: dict[str, list[str]] = defaultdict(list)
    for key in all_keys:
        if args.group_final_components > key.count("/"):
            raise ValueError(
                f"--group-final-components {args.group_final_components} must leave at least "
                f"one component in key {key!r}"
            )
        episode = key.rsplit("/", args.group_final_components)[0]
        episode_to_keys[episode].append(key)

    episode_to_keys = dict(list(episode_to_keys.items())[rank::world_size])
    my_keys = [key for keys in episode_to_keys.values() for key in keys]
    print(f"Rank {rank}: {len(all_keys)} total keys, {len(my_keys)} assigned")
    print(
        f"Rank {rank}: {len(episode_to_keys)} groups "
        f"(group-final-components={args.group_final_components}), {len(my_keys)} keys to process"
    )

    # Load model & tokenizer
    from transformers import AutoModel, AutoTokenizer

    model_subfolder = args.model_subfolder or None
    tokenizer_subfolder = args.tokenizer_subfolder or None

    print(f"Loading text model: {args.model}" + (f" (subfolder={model_subfolder})" if model_subfolder else "") + " ...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        subfolder=tokenizer_subfolder,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model,
        subfolder=model_subfolder,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = model.to(device).eval()
    print(f"Model loaded — hidden size: {model.config.hidden_size}")

    # Encode each caption at most once. Caption preparation stays ordered while
    # key-level marker checks and I/O run concurrently across group boundaries.
    caption_cache: dict[str, torch.Tensor] = {}

    def get_embed(caption):
        if caption not in caption_cache:
            embeds = encode_texts(
                [caption],
                model,
                tokenizer,
                device,
                max_length=args.max_length,
                num_hidden_layers_to_skip=args.num_hidden_layers_to_skip,
                crop_start=args.crop_start,
            )
            caption_cache[caption] = embeds[0].to(storage_dtype)
        return caption_cache[caption]

    pbar = tqdm(total=len(my_keys), desc=f"Rank {rank}", unit="key")
    counts = _process_episodes(episode_to_keys, get_embed, args, pbar)
    pbar.close()

    if counts["skipped"] > 0:
        print(f"Rank {rank}: skipped {counts['skipped']} already-completed keys")
    if counts["no_caption"] > 0:
        print(f"Rank {rank}: {counts['no_caption']} keys with no caption (written without text_embed)")
    if counts["failed"] > 0:
        print(f"Rank {rank}: {counts['failed']} keys failed")
        raise RuntimeError(f"Rank {rank} failed to write {counts['failed']} keys")
    print(f"Rank {rank}: done — {counts['written']} keys written")


if __name__ == "__main__":
    main()
