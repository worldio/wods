"""Split WODS videos into fixed-size, non-overlapping frame chunks.

Each input shard ``<view>`` becomes one full non-video metadata shard plus
video-only frame shards.  For example, with ``--chunk-frames 16``::

    exterior_image_1_left

becomes::

    meta/exterior_image_1_left
    frame_00000/exterior_image_1_left
    frame_00016/exterior_image_1_left
    frame_00032/exterior_image_1_left

All fields except video remain whole in ``meta/<view>``. Videos are decoded and
re-encoded because an arbitrary frame boundary cannot generally be copied from
the source MP4. Chunk timestamps are retained to preserve the source frame rate.

The script supports local and fsspec paths, including ``gs://``.  Independent
processes can split the key list with the standard ``RANK`` and ``WORLD_SIZE``
environment variables. Source shard reads retry forever after a 120-second
per-attempt timeout (override with ``WODS_FSSPEC_ATTEMPT_TIMEOUT_S``).

Usage:
  PYTHONUNBUFFERED=1 uv run python scripts/split_wods_to_chunks.py \
      /input/wods /output/wods --chunk-frames 16 --crf 16
"""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing
import os
import random
import signal
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TypeVar

# FFmpeg children immediately exec and never use gRPC; its fork hooks can corrupt their inherited file descriptors.
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

import fsspec
import tqdm

from wods.v2 import (
    list_wods_keys,
    list_wods_shards,
    read_wods_metadata,
    read_wods_shard_batched,
    wods_key_exists,
    write_wods_key_done,
    write_wods_metadata,
    write_wods_shard,
    write_wods_shard_done,
)

_worker_config = None
_worker_input_fs = None
T = TypeVar("T")


class FsspecOperationTimeout(TimeoutError):
    pass


@dataclass(frozen=True)
class FsspecRetryConfig:
    attempt_timeout_s: float = 120.0
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0
    jitter_s: float = 1.0

    @classmethod
    def from_env(cls) -> "FsspecRetryConfig":
        return cls(
            attempt_timeout_s=float(os.getenv("WODS_FSSPEC_ATTEMPT_TIMEOUT_S", "120")),
            initial_backoff_s=float(os.getenv("WODS_FSSPEC_INITIAL_BACKOFF_S", "1")),
            max_backoff_s=float(os.getenv("WODS_FSSPEC_MAX_BACKOFF_S", "60")),
            jitter_s=float(os.getenv("WODS_FSSPEC_JITTER_S", "1")),
        )


def retry_fsspec_operation_forever(
    operation: Callable[[], T],
    *,
    reset: Callable[[], None],
    description: str,
    config: FsspecRetryConfig,
) -> T:
    attempt = 0
    backoff = config.initial_backoff_s
    while True:
        attempt += 1
        try:
            with _operation_deadline(config.attempt_timeout_s):
                return operation()
        except _permanent_errors():
            raise
        except BaseException as error:
            if not _is_retryable_fsspec_error(error):
                raise
            reset()
            sleep = backoff + random.random() * config.jitter_s
            print(
                f"    Retrying fsspec read: {description} attempt={attempt} "
                f"sleep={sleep:.1f}s error={error!r}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep)
            backoff = min(backoff * 2, config.max_backoff_s)


@contextlib.contextmanager
def _operation_deadline(seconds: float):
    if (
        not seconds
        or os.name != "posix"
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def handle_timeout(_signum, _frame):
        raise FsspecOperationTimeout(f"fsspec operation exceeded {seconds:.1f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handle_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def _permanent_errors() -> tuple[type[BaseException], ...]:
    return FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError, KeyError


def _is_retryable_fsspec_error(error: BaseException) -> bool:
    if isinstance(error, (FsspecOperationTimeout, TimeoutError, OSError)):
        return True
    try:
        from fsspec.exceptions import FSTimeoutError

        if isinstance(error, FSTimeoutError):
            return True
    except Exception:
        pass
    if type(error).__module__.startswith(("aiohttp", "requests", "urllib3")):
        return True
    if isinstance(error, RuntimeError):
        message = str(error).lower()
        return any(
            token in message
            for token in (
                "loop is not running",
                "event loop is closed",
                "session is closed",
                "timeout",
                "fork-safe",
                "batched wods read failed",
            )
        )
    return False


def reset_worker_input_fs() -> None:
    global _worker_input_fs
    if _worker_input_fs is not None:
        with contextlib.suppress(Exception):
            _worker_input_fs.close()
        clear_cache = getattr(type(_worker_input_fs), "clear_instance_cache", None)
        if callable(clear_cache):
            with contextlib.suppress(Exception):
                clear_cache()
    assert _worker_config is not None
    input_path = _worker_config[0]
    _worker_input_fs, _, _ = fsspec.get_fs_token_paths(
        input_path,
        storage_options={"skip_instance_cache": True},
    )


def read_input_shard(input_path: str, key: str, shard: str) -> dict:
    def read():
        return read_wods_shard_batched(
            [input_path],
            [key],
            [shard],
            include_video=True,
            include_video_raw=False,
            include_depth=True,
            include_depth_raw=False,
            fs=_worker_input_fs,
        )[0]

    return retry_fsspec_operation_forever(
        read,
        reset=reset_worker_input_fs,
        description=f"{key}/{shard}",
        config=FsspecRetryConfig.from_env(),
    )


def setup_distributed() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must be in [0, {world_size}), got {rank}")
    return rank, world_size


def output_shard_name(shard: str, start: int) -> str:
    return f"frame_{start:05d}/{shard.lstrip('/')}"


def write_output_shard(output_path: str, key: str, shard: str, **data) -> None:
    write_wods_shard(output_path, key, shard, **data)
    write_wods_shard_done(output_path, key, shard)


def split_shard(
    input_path: str,
    output_path: str,
    key: str,
    shard: str,
    *,
    chunk_frames: int,
    crf: int,
    rgb: bool,
) -> int:
    """Split one source shard and return the number of chunks written."""
    data = read_input_shard(input_path, key, shard)
    item = f"{key}/{shard}"
    video = data.pop("video", None)
    if video is None:
        raise ValueError(f"{item} has no video")
    num_frames = len(video)
    if num_frames < 1:
        raise ValueError(f"{item} has no frames")

    timestamps = data.get("timestamps")
    if timestamps is not None and len(timestamps) != num_frames:
        raise ValueError(f"{item} has {num_frames} video frames but {len(timestamps)} timestamps")

    metadata_shard = f"meta/{shard.lstrip('/')}"
    write_output_shard(output_path, key, metadata_shard, **data)

    chunks_written = 0
    for start in range(0, num_frames, chunk_frames):
        end = min(start + chunk_frames, num_frames)
        output_shard = output_shard_name(shard, start)
        write_output_shard(
            output_path,
            key,
            output_shard,
            video=video[start:end],
            timestamps=timestamps[start:end] if timestamps is not None else None,
            crf=crf,
            rgb=rgb,
        )
        chunks_written += 1

    print(f"    {shard}: metadata + {num_frames} frames -> {chunks_written} video chunks", flush=True)
    return chunks_written


def split_key(
    input_path: str,
    output_path: str,
    key: str,
    *,
    chunk_frames: int,
    crf: int,
    rgb: bool,
) -> int:
    """Split every shard in one WODS key and return the total chunks written."""
    if wods_key_exists(output_path, key):
        print(f"  Skipping completed key: {key}", flush=True)
        return 0

    shards = list(list_wods_shards(input_path, key))
    if not shards:
        raise ValueError(f"{key} has no WODS shards")

    try:
        metadata = read_wods_metadata(input_path, key)
        caption = metadata.get("caption")
    except FileNotFoundError:
        caption = None
    write_wods_metadata(output_path, key, caption=caption)

    chunks_written = 0
    for shard in shards:
        chunks_written += split_shard(
            input_path,
            output_path,
            key,
            shard,
            chunk_frames=chunk_frames,
            crf=crf,
            rgb=rgb,
        )

    write_wods_key_done(output_path, key)
    return chunks_written


def setup_key_worker(input_path: str, output_path: str, chunk_frames: int, crf: int, rgb: bool) -> None:
    global _worker_config
    _worker_config = (input_path, output_path, chunk_frames, crf, rgb)
    reset_worker_input_fs()


def split_one_key(key: str) -> int:
    assert _worker_config is not None and _worker_input_fs is not None
    input_path, output_path, chunk_frames, crf, rgb = _worker_config
    print(f"  Processing key: {key}", flush=True)
    return split_key(
        input_path,
        output_path,
        key,
        chunk_frames=chunk_frames,
        crf=crf,
        rgb=rgb,
    )


def _canonical_path(path: str) -> tuple[type, str]:
    fs, _, paths = fsspec.get_fs_token_paths(path.rstrip("/"))
    return type(fs), paths[0].rstrip("/")


def split_dataset(args: argparse.Namespace) -> None:
    input_path = args.input.rstrip("/")
    output_path = args.output.rstrip("/")
    if _canonical_path(input_path) == _canonical_path(output_path):
        raise ValueError("input and output WODS paths must be different")

    rank, world_size = setup_distributed()
    all_keys = list(list_wods_keys(input_path))
    end = len(all_keys) if args.length is None else min(args.start + args.length, len(all_keys))
    selected_keys = all_keys[args.start:end]
    my_keys = selected_keys[rank::world_size]

    print(
        f"Rank {rank}/{world_size}: {len(all_keys)} total keys, "
        f"{len(selected_keys)} selected, {len(my_keys)} assigned",
        flush=True,
    )

    keys_written = 0
    chunks_written = 0
    failures: list[tuple[str, Exception]] = []
    with ProcessPoolExecutor(
        max_workers=args.key_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=setup_key_worker,
        initargs=(input_path, output_path, args.chunk_frames, args.crf, args.rgb),
    ) as executor:
        futures = {executor.submit(split_one_key, key): key for key in my_keys}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=f"Rank {rank}", unit="key"):
            key = futures[future]
            try:
                chunks = future.result()
                if chunks > 0:
                    keys_written += 1
                    chunks_written += chunks
            except Exception as error:
                failures.append((key, error))
                print(f"  ERROR {key}: {error}", file=sys.stderr, flush=True)

    print(
        f"Rank {rank}: wrote {keys_written} keys / {chunks_written} chunks; "
        f"{len(failures)} failures",
        flush=True,
    )
    if failures:
        failed_keys = ", ".join(key for key, _ in failures[:10])
        suffix = " ..." if len(failures) > 10 else ""
        raise RuntimeError(f"failed to split {len(failures)} keys: {failed_keys}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split WODS videos into fixed-size, non-overlapping frame chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output shard naming:
  <key>/meta/<source-shard>.*
  <key>/frame_00000/<source-shard>.video.mp4
  <key>/frame_00016/<source-shard>.video.mp4

After all distributed ranks finish, optionally create a fast key index with:
  uv run python scripts/write_key_list.py <output-wods>
        """,
    )
    parser.add_argument("input", help="Source WODS path")
    parser.add_argument("output", help="Separate output WODS path")
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=16,
        help="Maximum frames per output shard (default: 16)",
    )
    parser.add_argument("--crf", type=int, default=16, help="Video CRF for re-encoding (default: 16)")
    parser.add_argument("--key-workers", type=int, default=8, help="Keys processed concurrently (default: 8)")
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Use libx264rgb instead of the default YUV encoder",
    )
    parser.add_argument("--start", type=int, default=0, help="First source key index (default: 0)")
    parser.add_argument("--length", type=int, default=None, help="Maximum source keys to select")
    args = parser.parse_args()

    if args.chunk_frames < 1:
        parser.error("--chunk-frames must be >= 1")
    if args.key_workers < 1:
        parser.error("--key-workers must be >= 1")
    if args.start < 0:
        parser.error("--start must be >= 0")
    if args.length is not None and args.length < 0:
        parser.error("--length must be >= 0")
    return args


def main() -> None:
    split_dataset(parse_args())


if __name__ == "__main__":
    main()
