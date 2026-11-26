"""
Write the wods key list file for a wods dataset.

This script scans a wods dataset directory for all keys and writes
the key list.
"""

import argparse

from wods.v2 import list_wods_keys, write_wods_key_list


def main():
    parser = argparse.ArgumentParser(description="Write wods key list file for a wods dataset")
    parser.add_argument("path", help="Path to wods dataset")
    parser.add_argument("--max-workers", type=int, default=32, help="Maximum number of threads for parallel globbing")
    parser.add_argument(
        "--parallel-depth", type=int, default=0, help="Directory depth at which to parallelize (0 = disable)"
    )

    args = parser.parse_args()

    # List all keys by scanning for .key files
    keys = list(list_wods_keys(args.path, max_workers=args.max_workers, parallel_depth=args.parallel_depth))
    print(f"Found {len(keys)} keys")

    write_wods_key_list(args.path, keys)
    print(f"Wrote wods key list to {args.path}")


if __name__ == "__main__":
    main()
