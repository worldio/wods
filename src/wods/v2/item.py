import copy
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import av
import fsspec
import numpy as np
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

from .utils import crop_resize, read_depth_exr, write_depth_exr, write_video_rgb, write_video_yuv

CURRENT_VERSION = 2


def list_wods_keys(path: str, max_workers: int = 32, parallel_depth: int = 0):
    """
    Lists all keys in a wods dataset located at the specified path.

    Args:
        path: Root path of the wods dataset.
        max_workers: Maximum number of threads for parallel globbing.
        parallel_depth: Directory depth at which to parallelize. 0 means no
            parallelization (single recursive glob). 1 parallelizes over
            top-level subdirectories, 2 over second-level subdirectories, etc.
    """

    try:
        keys = read_wods_key_list(path)
        if keys:
            yield from keys
            return
    except FileNotFoundError:
        pass

    fs, token, paths = fsspec.get_fs_token_paths(path)
    clean_path = paths[0]

    if parallel_depth > 0:
        print(f"Using parallel globbing with max_workers={max_workers} and parallel_depth={parallel_depth}")

    # Walk down to the target depth, collecting .key files along the way
    key_files = []
    dirs_at_depth = [clean_path]
    for _ in range(parallel_depth):
        next_dirs = []
        for d in dirs_at_depth:
            try:
                entries = fs.ls(d, detail=False)
            except FileNotFoundError:
                continue
            for entry in entries:
                if entry.endswith(".key"):
                    key_files.append(entry)
                elif fs.isdir(entry):
                    next_dirs.append(entry)
        dirs_at_depth = next_dirs

    if parallel_depth > 0:
        print(
            f"Found {len(key_files)} keys at depth {parallel_depth} and {len(dirs_at_depth)} directories for parallel globbing"
        )

    # Glob each directory at the target depth in parallel
    def glob_subdir(subdir):
        return fs.glob(f"{subdir}/**/*.key")

    all_items = list(key_files)
    with ThreadPoolExecutor(max_workers=min(max_workers, max(len(dirs_at_depth), 1))) as pool:
        futures = {pool.submit(glob_subdir, d): d for d in dirs_at_depth}
        with tqdm(total=len(futures), desc="Listing keys", mininterval=10, disable=len(futures) <= 1) as pbar:
            for future in as_completed(futures):
                all_items.extend(future.result())
                pbar.update(1)

    for x in sorted(all_items):
        x = x.removeprefix(clean_path + "/")
        x = x[:-4]
        yield x


def list_wods_shards(path: str, key: str):
    """
    Lists all shards in a wods dataset located at the specified path and key.
    """

    fs, token, paths = fsspec.get_fs_token_paths(path)
    clean_path = paths[0] + "/" + key
    items = fs.glob(f"{path}/{key}/**/*.shr")

    for x in sorted(items):
        x = x.removeprefix(clean_path + "/")
        x = x[:-4]
        yield x


def read_wods_key_list(path: str) -> List[str]:
    """Reads the key list from a wods dataset located at the specified path."""
    with fsspec.open(f"{path}/.wods", "r") as f:
        data = json.load(f)

    version = data["version"]
    keys = data["keys"]  # List[str]

    assert version == CURRENT_VERSION, f"Unsupported wods version: {version}"

    return keys


def get_wods_video_stream(path: str, key: str, shard: str, mode: str = "rb"):
    return fsspec.open(f"{path}/{key}/{shard}.video.mp4", mode)


def read_wods_metadata(path: str, key: str):
    with fsspec.open(f"{path}/{key}.meta.json", "rb") as f:
        data = json.load(f)

    version = data["version"]
    caption = data["caption"]  # str

    assert version == CURRENT_VERSION, f"Unsupported wods version: {version}"

    return {
        "__key__": key,
        **({"caption": caption} if caption is not None else {}),
    }


def read_wods_video(
    path: str,
    key: str,
    shard: str,
    size: Optional[Tuple[int, int]] = None,
    intrinsics: Optional[torch.Tensor] = None,
    include_video: bool = True,
    include_video_raw: bool = False,
):
    if include_video_raw:
        assert size is None

    result = {}

    try:
        with get_wods_video_stream(path, key, shard, "rb") as f:
            video_bytes = f.read()

        if include_video_raw:
            assert size is None
            result["video_raw"] = video_bytes
        if include_video:
            video = _decode_video_pyav(video_bytes)
            if size is not None:
                video, intrinsics = crop_resize(video, size, intrinsics)
            result["video"] = video

    except FileNotFoundError:
        if size is not None:
            return {}  # Cannot determine intrinsics after resizing since we don't know the video size

    # Add intrinsics if provided
    if intrinsics is not None:
        result["intrinsics"] = intrinsics

    return result


def read_wods_timestamps(path: str, key: str, shard: str):
    try:
        with fsspec.open(f"{path}/{key}/{shard}.time.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}

    version = data["version"]
    assert version == CURRENT_VERSION, f"Unsupported wods version: {version}"

    timestamps = torch.tensor(data["timestamps"]) if data["timestamps"] is not None else None  # T

    return {
        **({"timestamps": timestamps} if timestamps is not None else {}),
    }


def read_wods_latents(path: str, key: str, shard: str):
    try:
        with fsspec.open(f"{path}/{key}/{shard}.latents.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}

    version = data["version"]
    assert version == CURRENT_VERSION, f"Unsupported wods version: {version}"

    latents = data["latents"] if data["latents"] is not None else None

    return {
        **({"latents": latents} if latents is not None else {}),
    }


def read_wods_camera(path: str, key: str, shard: str):
    try:
        with fsspec.open(f"{path}/{key}/{shard}.cam.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}

    version = data["version"]
    assert version == CURRENT_VERSION, f"Unsupported wods version: {version}"

    intrinsics = torch.tensor(data["intrinsics"]) if data["intrinsics"] is not None else None  # T(3)(3)
    c2w = torch.tensor(data["c2w"]) if data["c2w"] is not None else None  # T(3)(4)

    return {
        **({"intrinsics": intrinsics} if intrinsics is not None else {}),
        **({"c2w": c2w} if c2w is not None else {}),
    }


def read_wods_points(path: str, key: str, shard: str):
    try:
        with fsspec.open(f"{path}/{key}/{shard}.pts.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}

    version = data["version"]
    assert version == CURRENT_VERSION, f"Unsupported wods version: {version}"

    point_tracks = data["point_tracks"].to_dense().float() if data["point_tracks"] is not None else None  # TN3
    point_visibility = (
        data["point_visibility"].to_dense().float() if data["point_visibility"] is not None else None
    )  # TN1

    return {
        **({"point_tracks": point_tracks} if point_tracks is not None else {}),
        **({"point_visibility": point_visibility} if point_visibility is not None else {}),
    }


def read_wods_depth(path: str, key: str, shard: str, include_depth_raw: bool = False, include_depth: bool = True):
    result = {}
    try:
        with fsspec.open(f"{path}/{key}/{shard}.depth.exr", "rb") as f:
            data = f.read()

        if include_depth_raw:
            result["depth_raw"] = data
        if include_depth:
            depth = read_depth_exr(io.BytesIO(data))  # THW
            result["depth"] = depth.clone()
    except FileNotFoundError:
        return {}

    return result


def read_wods_proprio(path: str, key: str, shard: str):
    """Read proprioception data.

    Returns dict with 'proprio' key: (T, proprio_dim) float32 tensor.
    """
    try:
        with fsspec.open(f"{path}/{key}/{shard}.proprio.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}

    version = data["version"]
    assert version == CURRENT_VERSION, f"Unsupported wods version: {version}"

    proprio = data["proprio"] if data["proprio"] is not None else None

    return {
        **({"proprio": proprio} if proprio is not None else {}),
    }


def read_wods_action(path: str, key: str, shard: str):
    """Read action data.

    Returns dict with 'action' key: (T, action_dim) float32 tensor.
    """
    try:
        with fsspec.open(f"{path}/{key}/{shard}.action.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}

    version = data["version"]
    assert version == CURRENT_VERSION, f"Unsupported wods version: {version}"

    action = data["action"] if data["action"] is not None else None

    return {
        **({"action": action} if action is not None else {}),
    }


def read_wods_target(path: str, key: str, shard: str):
    """Read target data.

    Returns dict with 'target' key: (T, target_dim) float32 tensor.
    """
    try:
        with fsspec.open(f"{path}/{key}/{shard}.target.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}

    version = data["version"]
    assert version == CURRENT_VERSION, f"Unsupported wods version: {version}"

    target = data["target"] if data["target"] is not None else None

    return {
        **({"target": target} if target is not None else {}),
    }


def read_wods_text_embed(path: str, key: str, shard: str):
    """Read pre-computed text embeddings.

    Returns dict with 'text_embed' key: arbitrary tensor (e.g. (embed_dim,) or (L, embed_dim)).
    """
    try:
        with fsspec.open(f"{path}/{key}/{shard}.text_embed.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}

    version = data["version"]
    assert version == CURRENT_VERSION, f"Unsupported wods version: {version}"

    text_embed = data["text_embed"] if data["text_embed"] is not None else None

    return {
        **({"text_embed": text_embed} if text_embed is not None else {}),
    }


def read_wods_node_geometry(path: str, key: str, shard: str):
    """Read precomputed transfer_query per-node geometry (center+axis projected to this view).

    Returns dict with 'node_geometry' key: (T, N, P, 3) float tensor — per node, P anchor points
    (center, axis_tip), as (x, y, depth) in this shard's camera.
    """
    try:
        with fsspec.open(f"{path}/{key}/{shard}.node_geometry.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}
    assert data["version"] == CURRENT_VERSION, f"Unsupported wods version: {data['version']}"
    node_geometry = data["node_geometry"] if data["node_geometry"] is not None else None
    return {**({"node_geometry": node_geometry} if node_geometry is not None else {})}


def read_wods_node_graph(path: str, key: str, shard: str):
    """Read the static per-embodiment node graph (NOT per-frame).

    Returns the static node graph with edges, axis_mask, joint_type, and node_names.
    """
    try:
        with fsspec.open(f"{path}/{key}/{shard}.node_graph.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}
    assert data["version"] == CURRENT_VERSION, f"Unsupported wods version: {data['version']}"
    return {"node_graph": {"edges": data["edges"], "axis_mask": data["axis_mask"],
                           "joint_type": data["joint_type"], "node_names": data["node_names"]}}


def read_wods_node_action_geometry(path: str, key: str, shard: str):
    """Read the action-TARGET per-node geometry (where the nodes go under the executed action).

    Returns dict with 'node_action_geometry' key: (T, N, P, 3) float tensor — same format as
    node_geometry (per node, P anchor points (center, axis_tip) as a 3D point in this shard's camera).
    """
    try:
        with fsspec.open(f"{path}/{key}/{shard}.node_action_geometry.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}
    assert data["version"] == CURRENT_VERSION, f"Unsupported wods version: {data['version']}"
    nag = data["node_action_geometry"] if data["node_action_geometry"] is not None else None
    return {**({"node_action_geometry": nag} if nag is not None else {})}


def read_wods_node_pose(path: str, key: str, shard: str):
    """Read the per-node pose track (observation side, paired with node_geometry).

    Returns dict with 'node_pose' key: (T, N, 3, 4) float tensor — per frame and node, the node's pose at the
    recorded state as an SE(3) [R | t] taking the node's local coordinates into this shard's camera frame.
    Nodes without a pose are all zero.
    """
    try:
        with fsspec.open(f"{path}/{key}/{shard}.node_pose.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}
    assert data["version"] == CURRENT_VERSION, f"Unsupported wods version: {data['version']}"
    tcp = data["node_pose"] if data["node_pose"] is not None else None
    return {**({"node_pose": tcp} if tcp is not None else {})}


def read_wods_node_action_pose(path: str, key: str, shard: str):
    """Read the per-node commanded action pose track.

    Returns dict with 'node_action_pose' key: (T, N, 3, 4) float tensor — per frame and node, the node's pose at
    the joint configuration the row's action commands, as an SE(3) [R | t] taking the node's local coordinates
    into the world frame (the same track is stored in every shard). Nodes without a pose are all zero. Row i
    aligns with ``action[i]`` and ``node_action_geometry[i]``.
    """
    try:
        with fsspec.open(f"{path}/{key}/{shard}.node_action_pose.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}
    assert data["version"] == CURRENT_VERSION, f"Unsupported wods version: {data['version']}"
    tcp = data["node_action_pose"] if data["node_action_pose"] is not None else None
    return {**({"node_action_pose": tcp} if tcp is not None else {})}


def read_wods_node_proprio(path: str, key: str, shard: str):
    """Read precomputed transfer_query per-node proprio (each joint's centered qpos on its node).

    Returns dict with 'node_proprio' key: (T, N) float tensor — per node, the centered qpos of that
    node's joint (arm nodes 1:1, gripper drivers from their driver qpos; terminals 0).
    """
    try:
        with fsspec.open(f"{path}/{key}/{shard}.node_proprio.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}
    assert data["version"] == CURRENT_VERSION, f"Unsupported wods version: {data['version']}"
    node_proprio = data["node_proprio"] if data["node_proprio"] is not None else None
    return {**({"node_proprio": node_proprio} if node_proprio is not None else {})}


def read_wods_mask(path: str, key: str, shard: str):
    try:
        with fsspec.open(f"{path}/{key}/{shard}.mask.pt", "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return {}

    version = data["version"]
    assert version == CURRENT_VERSION, f"Unsupported wods version: {version}"

    rle_masks_dict = data["mask"] if data["mask"] is not None else None

    if rle_masks_dict is None:
        return {}

    # Decode RLE masks to binary masks for each named mask
    masks_dict = {}
    for name, rle_masks in rle_masks_dict.items():
        masks = []
        for rle in rle_masks:
            binary_mask = mask_utils.decode(rle)  # HW
            masks.append(binary_mask)
        mask = torch.tensor(np.stack(masks, axis=0), dtype=torch.bool)  # THW
        masks_dict[name] = mask

    return {"mask": masks_dict}


def read_wods_shard(
    path: str,
    key: str,
    shard: str,
    *,
    include_video: bool = True,
    include_video_raw: bool = False,
    include_timestamps: bool = True,
    include_latents: bool = True,
    include_camera: bool = True,
    include_points: bool = True,
    include_depth: bool = True,
    include_depth_raw: bool = False,
    include_mask: bool = True,
    include_proprio: bool = True,
    include_action: bool = True,
    include_target: bool = True,
    include_text_embed: bool = True,
    include_node_geometry: bool = True,
    include_node_action_geometry: bool = True,
    include_node_graph: bool = True,
    include_node_proprio: bool = True,
    include_node_action_pose: bool = True,
    include_node_pose: bool = True,
):
    item = {}
    if include_video or include_video_raw:
        item.update(read_wods_video(path, key, shard, include_video_raw=include_video_raw, include_video=include_video))
    if include_timestamps:
        item.update(read_wods_timestamps(path, key, shard))
    if include_latents:
        item.update(read_wods_latents(path, key, shard))
    if include_camera:
        item.update(read_wods_camera(path, key, shard))
    if include_points:
        item.update(read_wods_points(path, key, shard))
    if include_depth or include_depth_raw:
        item.update(read_wods_depth(path, key, shard, include_depth_raw=include_depth_raw, include_depth=include_depth))
    if include_mask:
        item.update(read_wods_mask(path, key, shard))
    if include_proprio:
        item.update(read_wods_proprio(path, key, shard))
    if include_action:
        item.update(read_wods_action(path, key, shard))
    if include_target:
        item.update(read_wods_target(path, key, shard))
    if include_text_embed:
        item.update(read_wods_text_embed(path, key, shard))
    if include_node_geometry:
        item.update(read_wods_node_geometry(path, key, shard))
    if include_node_action_geometry:
        item.update(read_wods_node_action_geometry(path, key, shard))
    if include_node_graph:
        item.update(read_wods_node_graph(path, key, shard))
    if include_node_proprio:
        item.update(read_wods_node_proprio(path, key, shard))
    if include_node_action_pose:
        item.update(read_wods_node_action_pose(path, key, shard))
    if include_node_pose:
        item.update(read_wods_node_pose(path, key, shard))
    return item


def read_wods_shard_batched(
    paths: List[str],
    keys: List[str],
    shards: List[str],
    *,
    include_video: bool = True,
    include_video_raw: bool = False,
    include_timestamps: bool = True,
    include_latents: bool = True,
    include_camera: bool = True,
    include_points: bool = True,
    include_depth: bool = True,
    include_depth_raw: bool = False,
    include_mask: bool = True,
    include_proprio: bool = True,
    include_action: bool = True,
    include_target: bool = True,
    include_text_embed: bool = True,
    include_node_geometry: bool = True,
    include_node_action_geometry: bool = True,
    include_node_graph: bool = True,
    include_node_proprio: bool = True,
    include_node_action_pose: bool = True,
    include_node_pose: bool = True,
    video_size: Optional[Tuple[int, int]] = None,
    fs=None,
) -> List[dict]:
    """Batch-fetch all requested files for multiple (path, key, shard) triples
    in a single ``fs.cat()`` call.

    Functionally equivalent to calling :func:`read_wods_shard` for each triple,
    but issues **one** async I/O dispatch for all files instead of sequential
    round-trips.  On remote filesystems the speedup is significant because the
    per-request latency (~100 ms on GCS) is paid once, not per file.

    All *paths* must resolve to the same fsspec filesystem type.

    Args:
        paths: Dataset root paths, one per sample (same length as *keys*/*shards*).
        keys: Sample keys, one per sample.
        shards: Shard names, one per sample.
        include_video / include_video_raw / ...: Same flags as :func:`read_wods_shard`.
        video_size: Optional ``(H, W)`` to resize video frames.
        fs: Pre-created fsspec filesystem (reuse across calls to avoid
            re-creation overhead).  Created automatically from *paths[0]* if
            ``None``.

    Returns:
        List of dicts (one per sample), each with the same schema as
        :func:`read_wods_shard`.
    """
    n = len(paths)
    assert (
        len(keys) == n and len(shards) == n
    ), f"paths/keys/shards must have the same length, got {len(paths)}/{len(keys)}/{len(shards)}"
    if n == 0:
        return []

    # -- Resolve filesystem and protocol-stripped prefixes -------------------
    unique_paths = sorted(set(paths))
    prefix_map: dict[str, str] = {}
    for p in unique_paths:
        fs_p, _, resolved = fsspec.get_fs_token_paths(p)
        if fs is None:
            fs = fs_p
        elif type(fs_p) is not type(fs):
            raise ValueError(
                f"All paths must use the same filesystem type, " f"got {type(fs).__name__} and {type(fs_p).__name__}"
            )
        prefix_map[p] = resolved[0]

    # -- Build the full list of remote paths to fetch -----------------------
    all_fetch_paths: list[str] = []
    path_info: dict[str, tuple[int, str]] = {}  # fetch_path -> (sample_idx, tag)

    for i in range(n):
        prefix = prefix_map[paths[i]]
        base = f"{prefix}/{keys[i]}/{shards[i]}"

        components: list[tuple[str, str]] = []
        if include_video or include_video_raw:
            components.append(("video", f"{base}.video.mp4"))
        if include_timestamps:
            components.append(("timestamps", f"{base}.time.json"))
        if include_latents:
            components.append(("latents", f"{base}.latents.pt"))
        if include_camera:
            components.append(("camera", f"{base}.cam.json"))
        if include_points:
            components.append(("points", f"{base}.pts.pt"))
        if include_depth or include_depth_raw:
            components.append(("depth", f"{base}.depth.exr"))
        if include_mask:
            components.append(("mask", f"{base}.mask.pt"))
        if include_proprio:
            components.append(("proprio", f"{base}.proprio.pt"))
        if include_action:
            components.append(("action", f"{base}.action.pt"))
        if include_target:
            components.append(("target", f"{base}.target.pt"))
        if include_text_embed:
            components.append(("text_embed", f"{base}.text_embed.pt"))
        if include_node_geometry:
            components.append(("node_geometry", f"{base}.node_geometry.pt"))
        if include_node_action_geometry:
            components.append(("node_action_geometry", f"{base}.node_action_geometry.pt"))
        if include_node_graph:
            components.append(("node_graph", f"{base}.node_graph.pt"))
        if include_node_proprio:
            components.append(("node_proprio", f"{base}.node_proprio.pt"))
        if include_node_action_pose:
            components.append(("node_action_pose", f"{base}.node_action_pose.pt"))
        if include_node_pose:
            components.append(("node_pose", f"{base}.node_pose.pt"))

        for tag, fp in components:
            all_fetch_paths.append(fp)
            path_info[fp] = (i, tag)

    if not all_fetch_paths:
        return [{} for _ in range(n)]

    # -- Single batched fetch -----------------------------------------------
    try:
        blobs = fs.cat(all_fetch_paths, on_error="return")
    except Exception as error:
        error.add_note("batched WODS read failed during fs.cat")
        raise

    # -- Decode per-sample --------------------------------------------------
    results: list[dict] = [{} for _ in range(n)]

    for fp in all_fetch_paths:
        if fp not in blobs:
            raise RuntimeError(f"batched WODS read returned no result for {fp}")
        data = blobs[fp]
        if isinstance(data, FileNotFoundError):
            continue
        if isinstance(data, Exception):
            data.add_note(f"batched WODS read failed for {fp}")
            raise data
        i, tag = path_info[fp]
        decoded = _decode_tag(
            tag,
            data,
            include_video=include_video,
            include_video_raw=include_video_raw,
            include_depth=include_depth,
            include_depth_raw=include_depth_raw,
            video_size=video_size,
        )
        results[i].update(decoded)

    return results


def _decode_video_pyav(data: bytes) -> torch.Tensor:
    """Decode MP4 bytes to a uint8 THWC tensor using PyAV.

    Calls libavformat/libavcodec in-process — ~20x faster than spawning an
    ffmpeg subprocess for single-frame or short videos.
    """
    container = av.open(io.BytesIO(data), format="mp4")
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    return torch.from_numpy(np.stack(frames))


def _decode_tag(
    tag: str,
    data: bytes,
    *,
    include_video: bool = True,
    include_video_raw: bool = False,
    include_depth: bool = True,
    include_depth_raw: bool = False,
    video_size: Optional[Tuple[int, int]] = None,
) -> dict:
    """Decode raw bytes for a given tag into a dict of output fields.

    This is the shared decode logic used by both ``read_wods_shard_batched``
    and the packed reader.
    """
    result = {}

    if tag == "video":
        if include_video_raw:
            result["video_raw"] = data
        if include_video:
            video = _decode_video_pyav(data)
            if video_size is not None:
                video, _ = crop_resize(video, video_size, None)
            result["video"] = video

    elif tag == "timestamps":
        ts_data = json.loads(data)
        assert ts_data["version"] == CURRENT_VERSION
        if ts_data["timestamps"] is not None:
            result["timestamps"] = torch.tensor(ts_data["timestamps"])

    elif tag == "latents":
        pt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert pt["version"] == CURRENT_VERSION
        if pt["latents"] is not None:
            result["latents"] = pt["latents"]

    elif tag == "camera":
        cam = json.loads(data)
        assert cam["version"] == CURRENT_VERSION
        if cam["intrinsics"] is not None:
            result["intrinsics"] = torch.tensor(cam["intrinsics"])
        if cam["c2w"] is not None:
            result["c2w"] = torch.tensor(cam["c2w"])

    elif tag == "points":
        pt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert pt["version"] == CURRENT_VERSION
        if pt["point_tracks"] is not None:
            result["point_tracks"] = pt["point_tracks"].to_dense().float()
        if pt["point_visibility"] is not None:
            result["point_visibility"] = pt["point_visibility"].to_dense().float()

    elif tag == "depth":
        if include_depth_raw:
            result["depth_raw"] = data
        if include_depth:
            result["depth"] = read_depth_exr(io.BytesIO(data)).clone()

    elif tag == "mask":
        mk = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert mk["version"] == CURRENT_VERSION
        rle_masks_dict = mk["mask"]
        if rle_masks_dict is not None:
            masks_dict = {}
            for name, rle_masks in rle_masks_dict.items():
                masks = [mask_utils.decode(rle) for rle in rle_masks]
                masks_dict[name] = torch.tensor(np.stack(masks, axis=0), dtype=torch.bool)
            result["mask"] = masks_dict

    elif tag == "proprio":
        pt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert pt["version"] == CURRENT_VERSION
        if pt["proprio"] is not None:
            result["proprio"] = pt["proprio"]

    elif tag == "action":
        pt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert pt["version"] == CURRENT_VERSION
        if pt["action"] is not None:
            result["action"] = pt["action"]

    elif tag == "target":
        pt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert pt["version"] == CURRENT_VERSION
        if pt["target"] is not None:
            result["target"] = pt["target"]

    elif tag == "text_embed":
        pt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert pt["version"] == CURRENT_VERSION
        if pt["text_embed"] is not None:
            result["text_embed"] = pt["text_embed"]

    elif tag == "node_geometry":
        pt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert pt["version"] == CURRENT_VERSION
        if pt["node_geometry"] is not None:
            result["node_geometry"] = pt["node_geometry"]

    elif tag == "node_action_geometry":
        pt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert pt["version"] == CURRENT_VERSION
        if pt["node_action_geometry"] is not None:
            result["node_action_geometry"] = pt["node_action_geometry"]

    elif tag == "node_graph":
        pt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert pt["version"] == CURRENT_VERSION
        result["node_graph"] = {"edges": pt["edges"], "axis_mask": pt["axis_mask"],
                                "joint_type": pt["joint_type"], "node_names": pt["node_names"]}

    elif tag == "node_proprio":
        pt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert pt["version"] == CURRENT_VERSION
        if pt["node_proprio"] is not None:
            result["node_proprio"] = pt["node_proprio"]

    elif tag == "node_action_pose":
        pt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert pt["version"] == CURRENT_VERSION
        if pt["node_action_pose"] is not None:
            result["node_action_pose"] = pt["node_action_pose"]

    elif tag == "node_pose":
        pt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        assert pt["version"] == CURRENT_VERSION
        if pt["node_pose"] is not None:
            result["node_pose"] = pt["node_pose"]

    elif tag == "metadata":
        meta = json.loads(data)
        assert meta["version"] == CURRENT_VERSION
        result["__key__"] = None  # caller fills in
        caption = meta.get("caption")
        if caption is not None:
            result["caption"] = caption

    return result


def write_wods_metadata(path: str, key: str, caption: str = None):
    info = {
        "version": CURRENT_VERSION,
        "caption": str(caption) if caption is not None else None,
    }
    with fsspec.open(f"{path}/{key}.meta.json", "w") as f:
        json.dump(info, f)


def write_wods_video_and_timestamps(
    path: str,
    key: str,
    shard: str,
    *,
    video: torch.Tensor = None,
    video_raw: bytes = None,
    timestamps: torch.Tensor = None,
    crf: int = 23,
    qp: int = None,
    rgb: bool = False,
):
    if video is not None and video_raw is not None:
        raise ValueError("Cannot specify both 'video' and 'video_raw' arguments")

    # Write video if either video or video_raw is provided
    if video is not None or video_raw is not None:
        fps = 8.0
        if timestamps is not None and len(timestamps) >= 2:
            fps = min(1 / (timestamps[1].item() - timestamps[0].item()), 60.0)

        with get_wods_video_stream(path, key, shard, "wb") as f:
            if video_raw is not None:
                f.write(video_raw)
            elif rgb:
                write_video_rgb(f, video.numpy(force=True), fps=fps, crf=crf, qp=qp)
            else:
                write_video_yuv(f, video.numpy(force=True), fps=fps, crf=crf, qp=qp)

    info = {
        "version": CURRENT_VERSION,
        "timestamps": timestamps.tolist() if timestamps is not None else None,  # T
    }
    with fsspec.open(f"{path}/{key}/{shard}.time.json", "w") as f:
        json.dump(info, f)


def _compact(value):
    """``value`` with every tensor that is a view into a larger storage copied out of it. torch.save writes a tensor's
    whole storage, so a slice saved as it is (say, one chunk of a trajectory's tensor) would carry all of the tensor it
    was cut from. A tensor that owns exactly its data is saved as it is, without a copy."""
    if isinstance(value, torch.Tensor):
        if value.layout == torch.strided and value.untyped_storage().nbytes() > value.numel() * value.element_size():
            return value.clone(memory_format=torch.contiguous_format)
        return value
    if isinstance(value, dict):
        return {name: _compact(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_compact(item) for item in value)
    return value


def write_wods_latents(path: str, key: str, shard: str, *, latents: torch.Tensor = None):
    info = {
        "version": CURRENT_VERSION,
        "latents": latents.detach() if latents is not None else None,
    }
    with fsspec.open(f"{path}/{key}/{shard}.latents.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_camera(
    path: str,
    key: str,
    shard: str,
    *,
    intrinsics: torch.Tensor = None,
    c2w: torch.Tensor = None,
):
    info = {
        "version": CURRENT_VERSION,
        "intrinsics": intrinsics.tolist() if intrinsics is not None else None,  # T(3)(3)
        "c2w": c2w.tolist() if c2w is not None else None,  # T(3)(4)
    }
    with fsspec.open(f"{path}/{key}/{shard}.cam.json", "w") as f:
        json.dump(info, f)


def write_wods_points(
    path: str, key: str, shard: str, *, point_tracks: torch.Tensor = None, point_visibility: torch.Tensor = None
):
    info = {
        "version": CURRENT_VERSION,
        "point_tracks": (
            point_tracks.detach().cpu().clone().to(torch.float16).to_sparse(2) if point_tracks is not None else None
        ),  # TN3
        "point_visibility": (
            point_visibility.detach().cpu().clone().to(torch.float16).to_sparse(2)
            if point_visibility is not None
            else None
        ),  # TN1
    }
    with fsspec.open(f"{path}/{key}/{shard}.pts.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_depth(path: str, key: str, shard: str, *, depth: torch.Tensor = None, depth_raw: bytes = None):
    """Write depth data to EXR format."""
    if depth is not None and depth_raw is not None:
        raise ValueError("Cannot specify both 'depth' and 'depth_raw' arguments")

    if depth is not None:
        with fsspec.open(f"{path}/{key}/{shard}.depth.exr", "wb") as f:
            write_depth_exr(f, depth)
    elif depth_raw is not None:
        with fsspec.open(f"{path}/{key}/{shard}.depth.exr", "wb") as f:
            f.write(depth_raw)


def write_wods_proprio(path: str, key: str, shard: str, *, proprio: torch.Tensor = None):
    """Write proprioception data.

    Args:
        proprio: (T, proprio_dim) float tensor of robot proprioception per frame.
    """
    info = {
        "version": CURRENT_VERSION,
        "proprio": proprio.detach() if proprio is not None else None,
    }
    with fsspec.open(f"{path}/{key}/{shard}.proprio.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_action(path: str, key: str, shard: str, *, action: torch.Tensor = None):
    """Write action data.

    Args:
        action: (T, action_dim) float tensor of executed actions per frame.
    """
    info = {
        "version": CURRENT_VERSION,
        "action": action.detach() if action is not None else None,
    }
    with fsspec.open(f"{path}/{key}/{shard}.action.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_target(path: str, key: str, shard: str, *, target: torch.Tensor = None):
    """Write target data (e.g. goal state the action is aiming at).

    Args:
        target: (T, target_dim) float tensor of per-frame targets.
    """
    info = {
        "version": CURRENT_VERSION,
        "target": target.detach() if target is not None else None,
    }
    with fsspec.open(f"{path}/{key}/{shard}.target.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_text_embed(path: str, key: str, shard: str, *, text_embed: torch.Tensor = None):
    """Write pre-computed text embeddings.

    Args:
        text_embed: Arbitrary tensor, e.g. (embed_dim,) or (L, embed_dim).
    """
    info = {
        "version": CURRENT_VERSION,
        "text_embed": text_embed.detach() if text_embed is not None else None,
    }
    with fsspec.open(f"{path}/{key}/{shard}.text_embed.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_node_geometry(path: str, key: str, shard: str, *, node_geometry: torch.Tensor = None):
    """Write precomputed transfer_query per-node geometry (projected to this view).

    Args:
        node_geometry: (T, N, P, 3) float tensor — per node, (center, axis_tip), (x, y, depth).
    """
    info = {
        "version": CURRENT_VERSION,
        "node_geometry": node_geometry.detach() if node_geometry is not None else None,
    }
    with fsspec.open(f"{path}/{key}/{shard}.node_geometry.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_node_pose(path: str, key: str, shard: str, *, node_pose: torch.Tensor = None):
    """Write the per-node pose track (see read_wods_node_pose).

    Args:
        node_pose: (T, N, 3, 4) float tensor — per node the SE(3) [R | t] into this shard's camera frame.
    """
    info = {
        "version": CURRENT_VERSION,
        "node_pose": node_pose.detach() if node_pose is not None else None,
    }
    with fsspec.open(f"{path}/{key}/{shard}.node_pose.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_node_action_pose(path: str, key: str, shard: str, *, node_action_pose: torch.Tensor = None):
    """Write the per-node commanded action pose track (see read_wods_node_action_pose).

    Args:
        node_action_pose: (T, N, 3, 4) float tensor — per node the SE(3) [R | t] into the WORLD frame.
    """
    info = {
        "version": CURRENT_VERSION,
        "node_action_pose": node_action_pose.detach() if node_action_pose is not None else None,
    }
    with fsspec.open(f"{path}/{key}/{shard}.node_action_pose.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_node_graph(path: str, key: str, shard: str, *, edges=None, axis_mask=None,
                          joint_type=None, node_names=None):
    """Write the static per-embodiment node graph (written once per shard, no T dim)."""
    info = {
        "version": CURRENT_VERSION,
        "edges": edges,
        "axis_mask": axis_mask,
        "joint_type": joint_type,
        "node_names": node_names,
    }
    with fsspec.open(f"{path}/{key}/{shard}.node_graph.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_node_action_geometry(path: str, key: str, shard: str, *, node_action_geometry: torch.Tensor = None):
    """Write the action-TARGET per-node geometry (same format as node_geometry, at the action pose).

    Args:
        node_action_geometry: (T, N, P, 3) float tensor — per node, (center, axis_tip), camera-frame 3D.
    """
    info = {
        "version": CURRENT_VERSION,
        "node_action_geometry": node_action_geometry.detach() if node_action_geometry is not None else None,
    }
    with fsspec.open(f"{path}/{key}/{shard}.node_action_geometry.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_node_proprio(path: str, key: str, shard: str, *, node_proprio: torch.Tensor = None):
    """Write precomputed transfer_query per-node proprio (each joint's centered qpos on its node).

    Args:
        node_proprio: (T, N) float tensor — per node, the joint's centered qpos; terminals 0.
    """
    info = {
        "version": CURRENT_VERSION,
        "node_proprio": node_proprio.detach() if node_proprio is not None else None,
    }
    with fsspec.open(f"{path}/{key}/{shard}.node_proprio.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_mask(path: str, key: str, shard: str, *, mask: dict = None):
    """Write mask data using pycocotools RLE encoding, stored with torch.save.

    Args:
        mask: Dict from name (str) to torch.Tensor of shape THW or HW
    """
    rle_masks_dict = None
    if mask is not None:
        rle_masks_dict = {}
        for name, mask_tensor in mask.items():
            # Ensure mask is THW format
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)  # HW -> 1HW

            # Convert to numpy and encode each frame with pycocotools
            mask_np = mask_tensor.numpy(force=True).astype(np.uint8)
            rle_masks = []
            for t in range(mask_np.shape[0]):
                # Encode binary mask to RLE using pycocotools
                # mask_utils.encode expects Fortran-ordered array
                rle = mask_utils.encode(np.asfortranarray(mask_np[t]))
                rle_masks.append(rle)
            rle_masks_dict[name] = rle_masks

    info = {
        "version": CURRENT_VERSION,
        "mask": rle_masks_dict,  # Dict from name to List of RLE dicts
    }
    with fsspec.open(f"{path}/{key}/{shard}.mask.pt", "wb") as f:
        torch.save(_compact(info), f)


def write_wods_shard(
    path: str,
    key: str,
    shard: str,
    *,
    video: torch.Tensor = None,
    video_raw: bytes = None,
    timestamps: torch.Tensor = None,
    latents: torch.Tensor = None,
    intrinsics: torch.Tensor = None,
    c2w: torch.Tensor = None,
    point_tracks: torch.Tensor = None,
    point_visibility: torch.Tensor = None,
    depth: torch.Tensor = None,
    depth_raw: bytes = None,
    mask: dict = None,
    proprio: torch.Tensor = None,
    action: torch.Tensor = None,
    target: torch.Tensor = None,
    text_embed: torch.Tensor = None,
    node_geometry: torch.Tensor = None,
    node_action_geometry: torch.Tensor = None,
    node_graph: dict = None,
    node_proprio: torch.Tensor = None,
    node_action_pose: torch.Tensor = None,
    node_pose: torch.Tensor = None,
    crf: int = 23,
    qp: int = None,
    rgb: bool = False,
):
    if video is not None or video_raw is not None or timestamps is not None:
        write_wods_video_and_timestamps(
            path,
            key,
            shard,
            video=video,
            video_raw=video_raw,
            timestamps=timestamps,
            crf=crf,
            qp=qp,
            rgb=rgb,
        )
    if latents is not None:
        write_wods_latents(path, key, shard, latents=latents)
    if intrinsics is not None or c2w is not None:
        write_wods_camera(path, key, shard, intrinsics=intrinsics, c2w=c2w)
    if point_tracks is not None or point_visibility is not None:
        write_wods_points(path, key, shard, point_tracks=point_tracks, point_visibility=point_visibility)
    if depth is not None or depth_raw is not None:
        write_wods_depth(path, key, shard, depth=depth, depth_raw=depth_raw)
    if mask is not None:
        write_wods_mask(path, key, shard, mask=mask)
    if proprio is not None:
        write_wods_proprio(path, key, shard, proprio=proprio)
    if action is not None:
        write_wods_action(path, key, shard, action=action)
    if target is not None:
        write_wods_target(path, key, shard, target=target)
    if text_embed is not None:
        write_wods_text_embed(path, key, shard, text_embed=text_embed)
    if node_geometry is not None:
        write_wods_node_geometry(path, key, shard, node_geometry=node_geometry)
    if node_action_geometry is not None:
        write_wods_node_action_geometry(path, key, shard, node_action_geometry=node_action_geometry)
    if node_graph is not None:
        write_wods_node_graph(
            path, key, shard,
            edges=node_graph.get("edges"),
            axis_mask=node_graph.get("axis_mask"),
            joint_type=node_graph.get("joint_type"),
            node_names=node_graph.get("node_names"),
        )
    if node_proprio is not None:
        write_wods_node_proprio(path, key, shard, node_proprio=node_proprio)
    if node_action_pose is not None:
        write_wods_node_action_pose(path, key, shard, node_action_pose=node_action_pose)
    if node_pose is not None:
        write_wods_node_pose(path, key, shard, node_pose=node_pose)


def write_wods_shard_done(path: str, key: str, shard: str):
    """Mark a shard as complete by creating a .shr marker file."""
    with fsspec.open(f"{path}/{key}/{shard}.shr", "w") as f:
        pass


def wods_key_exists(path: str, key: str) -> bool:
    """Check whether a key marker (.key file) exists."""
    fs, _, paths = fsspec.get_fs_token_paths(path)
    return fs.exists(f"{paths[0]}/{key}.key")


def wods_shard_exists(path: str, key: str, shard: str) -> bool:
    """Check whether a shard marker (.shr file) exists."""
    fs, _, paths = fsspec.get_fs_token_paths(path)
    return fs.exists(f"{paths[0]}/{key}/{shard}.shr")


def write_wods_key_done(path: str, key: str):
    """Mark a key as complete by creating a .key marker file."""
    with fsspec.open(f"{path}/{key}.key", "w") as f:
        pass


def write_wods_key_list(path: str, all_keys: List[str]):
    """Write key list."""
    all_keys = sorted(all_keys)
    with fsspec.open(f"{path}/.wods", "w") as f:
        json.dump({"version": CURRENT_VERSION, "keys": all_keys}, f)
