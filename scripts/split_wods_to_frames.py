"""
Split WODS dataset videos into individual frame shards.

This script takes a WODS dataset and splits each video (shard) into separate shards,
each containing a single frame, named frame00, frame01, etc.
"""

import argparse
import sys
from pathlib import Path

from wods.v2 import (
    list_wods_keys,
    list_wods_shards,
    read_wods_metadata,
    read_wods_shard,
    write_wods_key_done,
    write_wods_metadata,
    write_wods_shard,
    write_wods_shard_done,
)


def split_wods_to_frames(input_dir: str, output_dir: str):
    """Split each video in the WODS dataset into single-frame shards."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_path.mkdir(parents=True, exist_ok=True)

    keys = list_wods_keys(str(input_path))

    for key in keys:
        print(f"Processing item: {key}")

        shards = list_wods_shards(str(input_path), key)
        metadata = read_wods_metadata(str(input_path), key)

        # Write metadata for the new item
        write_wods_metadata(str(output_path), key, metadata.get("caption", f"Split frames for {key}"))

        for shard in shards:
            print(f"  Splitting shard: {shard}")

            shard_data = read_wods_shard(str(input_path), key, shard)
            video = shard_data["video"]
            timestamps = shard_data["timestamps"]
            intrinsics = shard_data["intrinsics"]
            c2w = shard_data["c2w"]
            T = video.shape[0]

            for i in range(T):
                new_shard_name = f"frame{i:02d}"

                new_video = video[i : i + 1]
                new_timestamps = timestamps[i : i + 1]
                new_c2w = c2w[i : i + 1]

                write_wods_shard(
                    str(output_path),
                    key,
                    new_shard_name,
                    video=new_video,
                    timestamps=new_timestamps,
                    intrinsics=intrinsics,
                    c2w=new_c2w,
                    crf=16,
                    rgb=True,
                )

                write_wods_shard_done(str(output_path), key, new_shard_name)

        write_wods_key_done(str(output_path), key)
        print(f"  Completed item: {key}")


def main():
    parser = argparse.ArgumentParser(description="Split WODS dataset videos into individual frame shards")
    parser.add_argument("input_dir", help="Path to input WODS dataset directory")
    parser.add_argument("output_dir", help="Path to output WODS dataset directory")

    args = parser.parse_args()

    try:
        split_wods_to_frames(args.input_dir, args.output_dir)
        print("Successfully split all videos into frame shards.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
