# `wods`: World Datasets

A dataset format for world models.

## Installation

```bash
pip install wods
```

## Usage

### Reading `wods` Items

```python
from wods.v2 import read_wods_metadata, read_wods_shard

# Read metadata for an item
metadata = read_wods_metadata("/path/to/dataset", "item_0")
# Returns: {"__key__": "item_0", "caption": "..."}

# Read a complete shard
data = read_wods_shard("/path/to/dataset", "item_0", "shard_0")
# Returns a dictionary with available data:
# - "video": THW3 tensor, in uint8 (0-255)
# - "timestamps": T tensor, in seconds
# - "latents": arbitrary tensor, for model latent representations
# - "intrinsics": T(3)(3) tensor, camera intrinsic matrices
# - "c2w": T(3)(4) tensor, camera-to-world transformation matrices
# - "point_tracks": TN3 tensor, 3D coordinates of tracked points over time
# - "point_visibility": TN1 tensor, visibility flags for each point over time
# - "depth": THW tensor, depth values in meters (float32)
# - "mask": dict from name (str) to THW tensor, boolean masks for segmentation
# - "proprio": T(proprio_dim) tensor, robot proprioception per frame
# - "action": T(action_dim) tensor, executed actions per frame
# - "target": T(target_dim) tensor, per-frame targets (e.g. goal state the action aims at)
# - "text_embed": arbitrary tensor, pre-computed text encoder embeddings
# - "node_geometry": T(N)(P)(3) tensor, robot kinematic nodes in this view; per node, P anchor points (center, axis_tip) as a 3D point in the shard's camera frame (wods convention), at the CURRENT (proprioceptive) pose
# - "node_action_geometry": T(N)(P)(3) tensor, the ACTION-TARGET counterpart of node_geometry — the same nodes at the pose the executed action commands (where they will be), same format
# - "node_graph": dict {edges (E,2), axis_mask (N,), joint_type (N,), node_names [str]}, the static node connectivity for this embodiment (no T dim)
# - "node_proprio": T(N) tensor, per-node proprio scalar — each joint's centered qpos scattered onto its node (arm nodes 1:1, gripper drivers from their driver qpos; terminals 0)
# - "node_pose": T(N)(3)(4) tensor, per frame and node the node's SE(3) pose [R | t] taking its local frame into this shard's camera frame (wods convention), at the CURRENT (proprioceptive) pose; nodes without a pose are all zero (e.g. only the TCP node carries a frame)
# - "node_action_pose": T(N)(3)(4) tensor, the commanded action counterpart of node_pose — per node the SE(3) pose [R | t] at the joint configuration the row's action commands, taken into the world frame (like c2w; the same track is stored in every shard); row i aligns with action[i], nodes without a pose are all zero
```

### Selective Data Loading

You can control which data types to load using the `include_*` options:

```python
# Only load video and timestamps, skip everything else
data = read_wods_shard(
    "/path/to/dataset", "item_0", "shard_0",
    include_video=True,
    include_video_raw=False,  # Skip raw video bytes
    include_depth=False,      # Skip depth data
    include_depth_raw=False,  # Skip raw depth bytes
    include_latents=False,    # Skip latent representations
    include_mask=False,       # Skip mask data
)
```

Available options:
- `include_video=True`: Include decoded video as THW3 tensor
- `include_video_raw=False`: Include raw video bytes (useful for recompression avoidance)
- `include_depth=True`: Include decoded depth as THW tensor
- `include_depth_raw=False`: Include raw depth bytes
- `include_latents=True`: Include latent representations
- `include_mask=True`: Include segmentation masks
- `include_proprio=True`: Include proprioception data
- `include_action=True`: Include action data
- `include_target=True`: Include target data
- `include_text_embed=True`: Include pre-computed text embeddings
- `include_node_geometry=True`: Include per-frame robot kinematic-node geometry (camera-frame 3D) at the current pose
- `include_node_action_geometry=True`: Include the action-target node geometry (same format as node_geometry)
- `include_node_graph=True`: Include the static node graph (edges, axis_mask, joint_type, node_names)
- `include_node_proprio=True`: Include the per-node proprio scalar (each joint's centered qpos on its node)
- `include_node_pose=True`: Include the per-node SE(3) pose track (camera-frame [R | t]) at the current pose
- `include_node_action_pose=True`: Include the per-node action-target SE(3) pose track (world-frame [R | t])

Note: If you need to read a video and write the same video elsewhere, it's advisable to read the raw video to avoid recompression by setting `include_video_raw=True`. If you don't need the decoded video at all, setting `include_video=False` will exclude the `"video"` key from the returned dict.

### Writing `wods` Items

```python
from wods.v2 import write_wods_metadata, write_wods_shard, write_wods_shard_done, write_wods_key_done
import torch

# Write metadata for an item
write_wods_metadata("/path/to/dataset", "item_0", caption="Sample video sequence")

# Write a complete shard with all data
write_wods_shard(
    path="/path/to/dataset",
    key="item_0",
    shard="shard_0",
    video=torch.randint(0, 256, (10, 384, 672, 3), dtype=torch.uint8),  # THW3 tensor
    timestamps=torch.arange(10) / 8.0,  # T tensor, in seconds (8 fps example)
    latents=torch.randn(128),  # Arbitrary tensor for model latent representations
    intrinsics=torch.eye(3).unsqueeze(0).repeat(10, 1, 1),  # T(3)(3) tensor, camera intrinsics
    c2w=torch.eye(3, 4).unsqueeze(0).repeat(10, 1, 1),  # T(3)(4) tensor, camera-to-world transforms
    point_tracks=torch.rand(10, 100, 3),  # TN3 tensor, 3D coordinates of tracked points
    point_visibility=torch.randint(0, 2, (10, 100, 1)),  # TN1 tensor, visibility flags
    depth=torch.rand(10, 384, 672) * 50 + 0.5,  # THW tensor, depth values in meters
    mask={  # dict from name (str) to THW tensor, boolean masks for segmentation
        "object": torch.randint(0, 2, (10, 384, 672), dtype=torch.bool),
        "background": torch.randint(0, 2, (10, 384, 672), dtype=torch.bool),
    },
    proprio=torch.randn(10, 7),  # T(proprio_dim) tensor, robot proprioception
    action=torch.randn(10, 7),  # T(action_dim) tensor, executed actions
    target=torch.randn(10, 7),  # T(target_dim) tensor, per-frame targets (e.g. goal state)
    text_embed=torch.randn(512),  # Arbitrary tensor, pre-computed text embeddings
    node_geometry=torch.randn(10, 11, 2, 3),  # T(N)(P)(3), kinematic nodes in this view: per node (center, axis_tip) as a camera-frame 3D point, current pose
    node_action_geometry=torch.randn(10, 11, 2, 3),  # T(N)(P)(3), same as node_geometry but at the action-TARGET pose
    node_graph={  # static node connectivity for this embodiment (no T dim)
        "edges": torch.tensor([[0, 1], [1, 2]]),        # (E, 2) node-id pairs (the kinematic tree)
        "axis_mask": torch.ones(11, dtype=torch.bool),  # (N,) True = actuated node carries a direction point
        "joint_type": torch.tensor([1] * 7 + [2, 2, 0, 0]),  # (N,) terminal=0, revolute=1, prismatic=2
        "node_names": ["J1", "J2", "..."],              # length-N node labels
    },
    node_proprio=torch.randn(10, 11),  # T(N), per-node proprio scalar: each joint's centered qpos on its node, terminals 0
    node_pose=torch.randn(10, 11, 3, 4),  # T(N)(3)(4), per-node SE(3) [R | t] into the shard camera frame, current pose; unposed nodes zero
    node_action_pose=torch.randn(10, 11, 3, 4),  # T(N)(3)(4), per-node SE(3) [R | t] into the world frame, at the commanded action pose
    crf=23,  # Quality parameter (lower = higher quality)
    rgb=False,  # Use YUV encoding (default) or RGB
)

# Mark the shard as complete
write_wods_shard_done("/path/to/dataset", "item_0", "shard_0")

# Mark the entire item as complete
write_wods_key_done("/path/to/dataset", "item_0")
```

To write a raw video, use the `video_raw=` key instead of the `video=` key.

### Listing `wods`

```python
from wods.v2 import list_wods_keys, list_wods_shards

# List all keys in a dataset
keys = list(list_wods_keys("/path/to/dataset"))

# List all shards for a specific key
shards = list(list_wods_shards("/path/to/dataset", "item_0"))
```

## Data Format

Each wods item consists of multiple shards, where each shard contains:
- `{key}/{shard}.video.mp4`: Video file containing image sequences
- `{key}/{shard}.time.json`: Timestamps data
- `{key}/{shard}.latents.pt`: Latent representations (arbitrary tensor for model latent representations)
- `{key}/{shard}.cam.json`: Camera parameters (intrinsics, camera-to-world transforms)
- `{key}/{shard}.pts.pt`: Point tracking data
- `{key}/{shard}.depth.exr`: Depth data (floating-point values in EXR format)
- `{key}/{shard}.mask.pt`: Segmentation masks (RLE-encoded boolean masks, stored as dict from name to masks)
- `{key}/{shard}.proprio.pt`: Proprioception data (per-frame robot state)
- `{key}/{shard}.action.pt`: Action data (per-frame executed actions)
- `{key}/{shard}.target.pt`: Target data (per-frame target the action aims at, e.g. goal joint positions)
- `{key}/{shard}.text_embed.pt`: Pre-computed text encoder embeddings
- `{key}/{shard}.node_geometry.pt`: Per-frame robot kinematic-node geometry (camera-frame 3D, current pose; per node: center + axis anchor points)
- `{key}/{shard}.node_action_geometry.pt`: Per-frame node geometry at the action-target pose (same format as node_geometry)
- `{key}/{shard}.node_graph.pt`: Static per-embodiment node graph (edges, axis_mask, joint_type, node_names; no T dim)
- `{key}/{shard}.node_proprio.pt`: Per-node proprio scalars (T,N), the joint's centered qpos scattered onto its node; terminals 0
- `{key}/{shard}.node_pose.pt`: Per-node SE(3) pose track (T,N,3,4), [R | t] into the shard camera frame at the current pose; unposed nodes zero
- `{key}/{shard}.node_action_pose.pt`: Per-node commanded action SE(3) pose track (T,N,3,4), [R | t] into the world frame; row i aligns with action[i]
- `{key}/{shard}.shr`: Empty marker file indicating the shard is complete
- `{key}.meta.json`: Metadata including caption
- `{key}.key`: Empty marker file indicating the item is complete

## Error Handling

The wods functions may raise exceptions. It will inevitably happen with large datasets. Handle these appropriately in your code.

## Coordinate System

Positions and rotations describe local-to-world transformations.

positions: `+x` - right, `+y` - up, `+z` - forward

rotations: `+x` - pitch, looking up, `+y` - yaw, looking to the right, `+z` - roll, head to the right; applied in `x` then `y` then `z` order (`Rz Ry Rx` if doing right-to-left matrix multiplication)
