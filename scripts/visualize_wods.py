"""
Comprehensive visualization script for wods format data.
Visualizes all available fields including video, depth, masks, point tracks, camera poses, etc.

Key Features:
- Reads and visualizes all wods data fields
- Multiple visualization modes:
  * overview: shows all available data types in grid layout
  * video: just the video frames
  * depth: depth visualization with colormap
  * mask: segmentation masks overlay
  * points: point tracks with trails
  * camera: camera trajectory visualization
  * combined: video with overlays (depth, mask, points)
- Configurable visualization parameters
- Preserves original video FPS by inferring from timestamps
- Visualizes all keys/shards when not specified

Usage:
uv run scripts/visualize_wods.py \
    --input=/path/to/wods/dataset \
    --output=/path/to/output/directory \
    --mode=overview
"""

import argparse
import colorsys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from epipolar_verify import compute_fundametal, draw_epipolar
from rich import print

from wods.v2 import list_wods_keys, list_wods_shards, read_wods_metadata, read_wods_shard


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize all wods dataset fields")
    parser.add_argument(
        "input",
        type=str,
        help="Input wods dataset path",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output directory path. Files will be saved as {output}/{key}/{shard}_{mode}.mp4",
    )
    parser.add_argument(
        "--key",
        type=str,
        help="Specific key to visualize (if not provided, will visualize all keys)",
    )
    parser.add_argument(
        "--shard",
        type=str,
        help="Specific shard to visualize (if not provided, will visualize all shards for the key)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["overview", "video", "depth", "mask", "points", "camera", "combined"],
        default="overview",
        help="Visualization mode: overview (grid layout), video, depth, mask, points, camera, or combined",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback output video frame rate (used when timestamps are not available to infer original FPS)",
    )
    # Point tracking options
    parser.add_argument(
        "--point-radius",
        type=int,
        default=3,
        help="Base radius of point circles (used when depth scaling is disabled)",
    )
    parser.add_argument(
        "--disable-depth-scaling",
        action="store_true",
        help="Disable depth-based point size scaling (depth scaling is enabled by default)",
    )
    parser.add_argument(
        "--max-point-radius",
        type=int,
        default=8,
        help="Maximum radius for nearby points when depth scaling is enabled",
    )
    parser.add_argument(
        "--min-point-radius",
        type=int,
        default=2,
        help="Minimum radius for faraway points when depth scaling is enabled",
    )
    parser.add_argument(
        "--depth-scale-factor",
        type=float,
        default=2.0,
        help="Scaling factor for depth-based size calculation (higher = more sensitive to depth changes)",
    )
    parser.add_argument(
        "--no-trails",
        action="store_true",
        help="Disable point trails (trails are shown by default)",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=10,
        help="Length of trails (number of previous frames to show)",
    )
    parser.add_argument(
        "--trail-fade-mode",
        type=str,
        choices=["immediate", "gradual"],
        default="gradual",
        help="How trails disappear when points become invisible",
    )
    # Depth visualization options
    parser.add_argument(
        "--depth-colormap",
        type=str,
        default="plasma",
        help="Matplotlib colormap for depth visualization",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        help="Minimum depth value for colormap (auto-computed if not provided)",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        help="Maximum depth value for colormap (auto-computed if not provided)",
    )
    # Mask visualization options
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.5,
        help="Alpha value for mask overlay (supports multiple named masks)",
    )
    # Camera visualization options
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=1.0,
        help="Scale factor for camera trajectory visualization",
    )
    parser.add_argument(
        "--show-camera-path",
        action="store_true",
        help="Show camera trajectory path",
    )
    return parser.parse_args()


def generate_distinct_colors(n_colors):
    """Generate n distinct colors using HSV color space with golden ratio sampling"""
    colors = []
    golden_ratio = 0.618033988749  # (sqrt(5) - 1) / 2

    for i in range(n_colors):
        # Use golden ratio for maximally distant hue sampling
        hue = (i * golden_ratio) % 1.0
        saturation = 0.7 + 0.3 * (i % 2)  # Alternate between 0.7 and 1.0
        value = 0.8 + 0.2 * ((i // 2) % 2)  # Alternate between 0.8 and 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert to BGR for OpenCV (0-255 range)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


def infer_fps_from_timestamps(timestamps):
    """Infer FPS from timestamps tensor"""
    if timestamps is None or len(timestamps) < 2:
        return None

    # Calculate time differences between consecutive frames
    time_diffs = []
    for i in range(1, len(timestamps)):
        diff = (timestamps[i] - timestamps[i - 1]).item()
        if diff > 0:  # Only consider positive differences
            time_diffs.append(diff)

    if not time_diffs:
        return None

    # Use median time difference to be robust against outliers
    median_time_diff = torch.tensor(time_diffs).median().item()
    inferred_fps = 1.0 / median_time_diff

    return inferred_fps


def apply_transparency(color, alpha, background_color=(128, 128, 128)):
    """Apply transparency effect by blending color with background"""
    return tuple(int(color[i] * alpha + background_color[i] * (1 - alpha)) for i in range(3))


def calculate_point_radius(depth_value, args, depth_stats=None):
    """Calculate point radius based on depth value

    Args:
        depth_value: Depth value for the point
        args: Command line arguments
        depth_stats: Dict with 'min' and 'max' depth values for normalization

    Returns:
        Point radius as integer
    """
    if args.disable_depth_scaling or depth_value <= 0 or depth_stats is None:
        return args.point_radius

    # Normalize depth to [0, 1] range
    depth_min, depth_max = depth_stats["min"], depth_stats["max"]
    if depth_max <= depth_min:
        return args.point_radius

    # Normalize depth (0 = close, 1 = far)
    normalized_depth = (depth_value - depth_min) / (depth_max - depth_min)
    normalized_depth = max(0, min(1, normalized_depth))  # Clamp to [0, 1]

    # Apply exponential decay for more natural scaling
    # f(d) = min_radius + (max_radius - min_radius) * exp(-scale_factor * d)
    size_range = args.max_point_radius - args.min_point_radius
    radius = args.min_point_radius + size_range * np.exp(-args.depth_scale_factor * normalized_depth)

    return int(max(args.min_point_radius, min(args.max_point_radius, radius)))


def visualize_depth(depth_tensor, colormap="plasma", vmin=None, vmax=None):
    """Convert depth tensor to colored visualization"""
    if depth_tensor is None:
        return None

    # Convert to numpy
    depth_np = depth_tensor.numpy()

    # Handle invalid depths (set to NaN for proper colormap handling)
    depth_np = np.where(depth_np <= 0, np.nan, depth_np)

    # Auto-compute range if not provided
    if vmin is None:
        vmin = np.nanmin(depth_np)
    if vmax is None:
        vmax = np.nanmax(depth_np)

    # Normalize depth values
    depth_norm = (depth_np - vmin) / (vmax - vmin)
    depth_norm = np.clip(depth_norm, 0, 1)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored_depth = cmap(depth_norm)  # Returns RGBA

    # Convert to RGB and scale to 0-255
    colored_depth_rgb = (colored_depth[:, :, :3] * 255).astype(np.uint8)

    return colored_depth_rgb


def visualize_mask(mask_data):
    """Convert mask data to colored visualization (standalone, not overlay)

    Args:
        mask_data: Either a single tensor or dict from name to tensor
    """
    if mask_data is None:
        return None

    # Handle both old single tensor format and new dict format
    if isinstance(mask_data, dict):
        # New dictionary format - combine all masks with different colors
        if not mask_data:
            return None

        # Get dimensions from first mask
        first_mask = next(iter(mask_data.values()))
        H, W = first_mask.shape

        # Create colored mask visualization
        mask_vis = np.zeros((H, W, 3), dtype=np.uint8)

        # Generate distinct colors for each mask
        mask_names = list(mask_data.keys())
        colors = generate_distinct_colors(len(mask_names))

        # Combine all masks with different colors
        for i, (name, mask_tensor) in enumerate(mask_data.items()):
            mask_np = mask_tensor.numpy().astype(np.uint8)
            color = colors[i]

            # Apply color where mask is active
            for c in range(3):
                mask_vis[:, :, c] = np.where(mask_np > 0, color[2 - c], mask_vis[:, :, c])  # RGB order

        return mask_vis
    else:
        # Old single tensor format (for backward compatibility)
        mask_np = mask_data.numpy().astype(np.uint8)
        H, W = mask_np.shape

        # Create colored mask visualization (standalone)
        mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
        # Use red for masks
        mask_vis[:, :, 0] = mask_np * 255  # Red channel

        return mask_vis


def create_epipolar_visualization(video_tensor, c2w_tensor, intrinsics_tensor, frame_idx):
    """Create epipolar line visualization between first frame and current frame"""
    if c2w_tensor is None or intrinsics_tensor is None or video_tensor is None:
        return None

    if frame_idx == 0 or frame_idx >= len(c2w_tensor):
        # For first frame or invalid frame, just return the current frame
        if frame_idx < len(video_tensor):
            current_frame = video_tensor[frame_idx].numpy().astype(np.uint8)
            # Return side-by-side with itself to maintain aspect ratio
            return np.concatenate([current_frame, current_frame], axis=1)
        return None

    try:
        # Get first frame and current frame
        first_frame = video_tensor[0].numpy().astype(np.uint8).copy()
        current_frame = video_tensor[frame_idx].numpy().astype(np.uint8).copy()

        # Get camera poses
        first_c2w = c2w_tensor[0]  # 3 x 4 or 4 x 4
        current_c2w = c2w_tensor[frame_idx]  # 3 x 4 or 4 x 4

        # Convert to 4x4 if needed
        if first_c2w.shape[0] == 3:
            first_c2w_4x4 = torch.eye(4)
            first_c2w_4x4[:3, :] = first_c2w
            first_c2w = first_c2w_4x4

        if current_c2w.shape[0] == 3:
            current_c2w_4x4 = torch.eye(4)
            current_c2w_4x4[:3, :] = current_c2w
            current_c2w = current_c2w_4x4

        # Apply coordinate system adjustments (same as epipolar_verify.py)
        first_c2w_adjusted = first_c2w.clone()
        first_c2w_adjusted[1, :] *= -1
        first_c2w_adjusted[:, 1] *= -1

        current_c2w_adjusted = current_c2w.clone()
        current_c2w_adjusted[1, :] *= -1
        current_c2w_adjusted[:, 1] *= -1

        # Get intrinsics
        first_intrinsics = intrinsics_tensor[0] if intrinsics_tensor.ndim == 3 else intrinsics_tensor
        current_intrinsics = intrinsics_tensor[frame_idx] if intrinsics_tensor.ndim == 3 else intrinsics_tensor

        # Compute fundamental matrix
        F = compute_fundametal(
            first_c2w_adjusted.numpy(),
            first_intrinsics.numpy(),
            current_c2w_adjusted.numpy(),
            current_intrinsics.numpy(),
        )

        # Draw epipolar lines (modifies current_frame in place)
        draw_epipolar(first_frame, current_frame, F, num_points=6)

        # Return side-by-side visualization
        return np.concatenate([first_frame, current_frame], axis=1)

    except Exception as e:
        # Fallback: return side-by-side frames without epipolar lines
        first_frame = video_tensor[0].numpy().astype(np.uint8)
        current_frame = video_tensor[frame_idx].numpy().astype(np.uint8)
        return np.concatenate([first_frame, current_frame], axis=1)


def compute_depth_stats_for_points(point_tracks, point_visibility):
    """Compute min/max depth statistics for all visible points across all frames

    Args:
        point_tracks: Tensor of shape (T, N, 3) with x, y, depth
        point_visibility: Tensor of shape (T, N, 1) with visibility flags

    Returns:
        Dict with 'min' and 'max' depth values, or None if no valid depths
    """
    if point_tracks is None or point_visibility is None or point_tracks.shape[-1] < 3:
        return None

    # Get all depth values for visible points
    visible_mask = point_visibility[:, :, 0] > 0.5
    depth_values = point_tracks[:, :, 2][visible_mask]

    # Filter out invalid depths
    valid_depths = depth_values[depth_values > 0]

    if len(valid_depths) == 0:
        return None

    return {"min": valid_depths.min().item(), "max": valid_depths.max().item()}


def create_overview_frame(
    data: Dict[str, Any],
    args,
    frame_idx: int,
    point_history: Optional[List] = None,
    colors: Optional[List] = None,
    depth_stats: Optional[Dict] = None,
) -> np.ndarray:
    """Create overview frame with all available visualizations in a grid layout"""

    # Get main video frame
    video = data.get("video")
    if video is None:
        return None

    main_frame = video[frame_idx].numpy().astype(np.uint8)
    H, W, C = main_frame.shape

    # Prepare visualizations
    visualizations = []
    titles = []

    # 1. Original video
    visualizations.append(main_frame)
    titles.append("Video")

    # 2. Depth visualization
    depth = data.get("depth")
    if depth is not None and frame_idx < depth.shape[0]:
        depth_vis = visualize_depth(depth[frame_idx], args.depth_colormap, args.depth_min, args.depth_max)
        visualizations.append(depth_vis)
        titles.append("Depth")

    # 3. Mask visualization
    mask = data.get("mask")
    if mask is not None:
        if isinstance(mask, dict):
            # Dictionary format - get masks for current frame
            frame_masks = {}
            for name, mask_tensor in mask.items():
                if frame_idx < mask_tensor.shape[0]:
                    frame_masks[name] = mask_tensor[frame_idx]
            if frame_masks:
                mask_vis = visualize_mask(frame_masks)
                if mask_vis is not None:
                    visualizations.append(mask_vis)
                    titles.append("Masks")
        else:
            # Single tensor format (backward compatibility)
            if frame_idx < mask.shape[0]:
                mask_vis = visualize_mask(mask[frame_idx])
                if mask_vis is not None:
                    visualizations.append(mask_vis)
                    titles.append("Mask")

    # 4. Point tracks overlay
    point_tracks = data.get("point_tracks")
    point_visibility = data.get("point_visibility")
    if point_tracks is not None and point_visibility is not None and frame_idx < point_tracks.shape[0]:
        points_frame = main_frame.copy()
        # Convert RGB to BGR for OpenCV
        if C == 3:
            points_frame = cv2.cvtColor(points_frame, cv2.COLOR_RGB2BGR)

        current_tracks = point_tracks[frame_idx]
        current_visibility = point_visibility[frame_idx]

        N = current_tracks.shape[0]
        if colors is None:
            colors = generate_distinct_colors(N)

        # Draw points
        for i in range(N):
            if current_visibility[i, 0] > 0.5:  # Point is visible
                x, y = current_tracks[i, 0].item(), current_tracks[i, 1].item()
                x = max(0, min(W - 1, int(x)))
                y = max(0, min(H - 1, int(y)))

                # Draw trails if enabled and available
                if not args.no_trails and point_history and len(point_history[i]) > 1:
                    for j in range(1, len(point_history[i])):
                        pt1 = point_history[i][j - 1]
                        pt2 = point_history[i][j]
                        alpha = (j / len(point_history[i])) * 0.5
                        color = apply_transparency(colors[i], alpha)
                        cv2.line(points_frame, pt1, pt2, color, 2)

                # Calculate point radius based on depth
                if not args.disable_depth_scaling and current_tracks.shape[-1] >= 3:
                    depth_value = current_tracks[i, 2].item()
                    point_radius = calculate_point_radius(depth_value, args, depth_stats)
                else:
                    point_radius = args.point_radius

                # Draw point
                cv2.circle(points_frame, (x, y), point_radius, colors[i], -1)

        # Convert back to RGB
        if C == 3:
            points_frame = cv2.cvtColor(points_frame, cv2.COLOR_BGR2RGB)

        visualizations.append(points_frame)
        titles.append("Point Tracks")

    # 5. Epipolar verification (first frame to current frame)
    c2w = data.get("c2w")
    if c2w is not None and video is not None:
        epipolar_vis = create_epipolar_visualization(video, c2w, data.get("intrinsics"), frame_idx)
        if epipolar_vis is not None:
            # Resize to match video frame size (note: epipolar vis is 2x wide)
            epipolar_vis_resized = cv2.resize(epipolar_vis, (2 * W, H))
            visualizations.append(epipolar_vis_resized)
            titles.append("Epipolar (Frame 0 -> Current)")

    # Create grid layout
    n_vis = len(visualizations)
    if n_vis == 0:
        return main_frame

    # Determine grid dimensions and handle 2x wide panels
    if n_vis == 1:
        grid_rows, grid_cols = 1, 1
    elif n_vis <= 2:
        grid_rows, grid_cols = 1, 2
    elif n_vis <= 4:
        grid_rows, grid_cols = 2, 2
    elif n_vis <= 6:
        grid_rows, grid_cols = 2, 3
    else:
        grid_rows, grid_cols = 3, 3

    # Create grid canvas - account for epipolar panels being 2x wide
    title_height = 30
    cell_height = H + title_height
    cell_width = W

    # Calculate max width needed (some panels might be 2x wide)
    max_width_needed = 0
    for i, (vis, title) in enumerate(zip(visualizations, titles)):
        row = i // grid_cols
        col = i % grid_cols
        panel_width = vis.shape[1]  # Actual width of this panel
        row_width = (col * cell_width) + panel_width
        max_width_needed = max(max_width_needed, row_width)

    grid_height = grid_rows * cell_height
    grid_width = max(max_width_needed, grid_cols * cell_width)
    grid_canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Place visualizations in grid
    for i, (vis, title) in enumerate(zip(visualizations, titles)):
        if i >= grid_rows * grid_cols:
            break

        row = i // grid_cols
        col = i % grid_cols

        y_start = row * cell_height
        x_start = col * cell_width

        # Get actual panel dimensions
        panel_height, panel_width = vis.shape[:2]

        # Add title (width matches the panel)
        cv2.rectangle(
            grid_canvas, (x_start, y_start), (x_start + panel_width, y_start + title_height), (50, 50, 50), -1
        )
        cv2.putText(grid_canvas, title, (x_start + 5, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Add visualization (resize if needed to fit)
        if panel_height == H and panel_width <= grid_width - x_start:
            # Panel fits perfectly
            grid_canvas[y_start + title_height : y_start + cell_height, x_start : x_start + panel_width] = vis
        else:
            # Resize panel to fit
            available_width = min(panel_width, grid_width - x_start)
            vis_resized = cv2.resize(vis, (available_width, H))
            grid_canvas[y_start + title_height : y_start + cell_height, x_start : x_start + available_width] = (
                vis_resized
            )

    return grid_canvas


def visualize_single_mode(data: Dict[str, Any], args, mode: str) -> List[np.ndarray]:
    """Create frames for a specific visualization mode"""

    video = data.get("video")
    if video is None:
        print("No video data available")
        return []

    T, H, W, C = video.shape
    frames = []

    # Initialize point tracking data if needed
    point_tracks = data.get("point_tracks")
    point_visibility = data.get("point_visibility")
    colors = None
    point_history = None
    depth_stats = None

    if mode in ["points", "combined", "overview"] and point_tracks is not None:
        N = point_tracks.shape[1]
        colors = generate_distinct_colors(N)
        if not args.no_trails:
            point_history = [[] for _ in range(N)]

        # Compute depth statistics for point size scaling
        if not args.disable_depth_scaling:
            depth_stats = compute_depth_stats_for_points(point_tracks, point_visibility)
            if depth_stats:
                print(f"Depth range for point scaling: {depth_stats['min']:.3f} - {depth_stats['max']:.3f}")
            else:
                print("Warning: No valid depth data found for point scaling")

    for t in range(T):
        if mode == "video":
            frame = video[t].numpy().astype(np.uint8)

        elif mode == "depth":
            depth = data.get("depth")
            if depth is not None and t < depth.shape[0]:
                frame = visualize_depth(depth[t], args.depth_colormap, args.depth_min, args.depth_max)
            else:
                frame = np.zeros((H, W, 3), dtype=np.uint8)
                cv2.putText(frame, "No depth data", (W // 4, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        elif mode == "mask":
            mask = data.get("mask")
            if mask is not None:
                if isinstance(mask, dict):
                    # Dictionary format - get masks for current frame
                    frame_masks = {}
                    for name, mask_tensor in mask.items():
                        if t < mask_tensor.shape[0]:
                            frame_masks[name] = mask_tensor[t]
                    if frame_masks:
                        frame = visualize_mask(frame_masks)
                    else:
                        frame = None
                else:
                    # Single tensor format (backward compatibility)
                    if t < mask.shape[0]:
                        frame = visualize_mask(mask[t])
                    else:
                        frame = None

                if frame is None:
                    frame = np.zeros((H, W, 3), dtype=np.uint8)
                    cv2.putText(
                        frame, "No mask data", (W // 4, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                    )
            else:
                frame = np.zeros((H, W, 3), dtype=np.uint8)
                cv2.putText(frame, "No mask data", (W // 4, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        elif mode == "points":
            frame = video[t].numpy().astype(np.uint8)
            if point_tracks is not None and point_visibility is not None and t < point_tracks.shape[0]:
                # Convert RGB to BGR for OpenCV
                if C == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                current_tracks = point_tracks[t]
                current_visibility = point_visibility[t]

                # Update and draw points (similar to original point visualization logic)
                for i in range(current_tracks.shape[0]):
                    if current_visibility[i, 0] > 0.5:  # Point is visible
                        x, y = current_tracks[i, 0].item(), current_tracks[i, 1].item()
                        x = max(0, min(W - 1, int(x)))
                        y = max(0, min(H - 1, int(y)))

                        # Update point history for trails
                        if not args.no_trails:
                            point_history[i].append((x, y))
                            if len(point_history[i]) > args.trail_length:
                                point_history[i].pop(0)

                        # Draw trail if enabled and we have history (including current point)
                        if not args.no_trails and len(point_history[i]) > 1:
                            for j in range(1, len(point_history[i])):
                                pt1 = point_history[i][j - 1]
                                pt2 = point_history[i][j]
                                alpha = (j / len(point_history[i])) * 0.5
                                color = apply_transparency(colors[i], alpha)
                                cv2.line(frame, pt1, pt2, color, 2)

                        # Calculate point radius based on depth
                        if not args.disable_depth_scaling and current_tracks.shape[-1] >= 3:
                            depth_value = current_tracks[i, 2].item()
                            point_radius = calculate_point_radius(depth_value, args, depth_stats)
                        else:
                            point_radius = args.point_radius

                        # Draw point
                        cv2.circle(frame, (x, y), point_radius, colors[i], -1)

                    else:  # Point not visible - handle trails
                        if not args.no_trails:
                            if args.trail_fade_mode == "immediate":
                                point_history[i].clear()
                            elif args.trail_fade_mode == "gradual":
                                if len(point_history[i]) > 1:
                                    for j in range(1, len(point_history[i])):
                                        pt1 = point_history[i][j - 1]
                                        pt2 = point_history[i][j]
                                        alpha = (j / len(point_history[i])) * 0.3
                                        color = apply_transparency(colors[i], alpha)
                                        cv2.line(frame, pt1, pt2, color, 2)

                                if len(point_history[i]) > 0:
                                    point_history[i].pop(0)

                # Convert back to RGB
                if C == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        elif mode == "camera":
            c2w = data.get("c2w")
            if c2w is not None:
                epipolar_vis = create_epipolar_visualization(video, c2w, data.get("intrinsics"), t)
                if epipolar_vis is not None:
                    # Keep original size since epipolar vis is already 2x wide
                    frame = epipolar_vis
                else:
                    frame = np.zeros((H, 2 * W, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        "No camera/epipolar data",
                        (W // 2, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
            else:
                frame = np.zeros((H, 2 * W, 3), dtype=np.uint8)
                cv2.putText(frame, "No camera data", (W // 2, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        elif mode == "combined":
            # Combine video with available overlays
            frame = video[t].numpy().astype(np.uint8)

            # Add depth overlay (if available)
            depth = data.get("depth")
            if depth is not None and t < depth.shape[0]:
                depth_vis = visualize_depth(depth[t], args.depth_colormap, args.depth_min, args.depth_max)
                # Blend with alpha
                frame = cv2.addWeighted(frame, 0.7, depth_vis, 0.3, 0)

            # Add mask overlay (if available)
            mask = data.get("mask")
            if mask is not None:
                mask_overlay = None
                if isinstance(mask, dict):
                    # Dictionary format - get masks for current frame
                    frame_masks = {}
                    for name, mask_tensor in mask.items():
                        if t < mask_tensor.shape[0]:
                            frame_masks[name] = mask_tensor[t]
                    if frame_masks:
                        mask_overlay = visualize_mask(frame_masks)
                else:
                    # Single tensor format (backward compatibility)
                    if t < mask.shape[0]:
                        mask_overlay = visualize_mask(mask[t])

                if mask_overlay is not None:
                    frame = cv2.addWeighted(frame, 1 - args.mask_alpha, mask_overlay, args.mask_alpha, 0)

            # Add point tracks (if available)
            if point_tracks is not None and point_visibility is not None and t < point_tracks.shape[0]:
                # Convert RGB to BGR for OpenCV processing
                if C == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                current_tracks = point_tracks[t]
                current_visibility = point_visibility[t]

                # Same point drawing logic as "points" mode
                for i in range(current_tracks.shape[0]):
                    if current_visibility[i, 0] > 0.5:
                        x, y = current_tracks[i, 0].item(), current_tracks[i, 1].item()
                        x = max(0, min(W - 1, int(x)))
                        y = max(0, min(H - 1, int(y)))

                        if not args.no_trails:
                            point_history[i].append((x, y))
                            if len(point_history[i]) > args.trail_length:
                                point_history[i].pop(0)

                        if not args.no_trails and len(point_history[i]) > 1:
                            for j in range(1, len(point_history[i])):
                                pt1 = point_history[i][j - 1]
                                pt2 = point_history[i][j]
                                alpha = (j / len(point_history[i])) * 0.5
                                color = apply_transparency(colors[i], alpha)
                                cv2.line(frame, pt1, pt2, color, 2)

                        # Calculate point radius based on depth
                        if not args.disable_depth_scaling and current_tracks.shape[-1] >= 3:
                            depth_value = current_tracks[i, 2].item()
                            point_radius = calculate_point_radius(depth_value, args, depth_stats)
                        else:
                            point_radius = args.point_radius

                        cv2.circle(frame, (x, y), point_radius, colors[i], -1)

                    else:
                        if not args.no_trails:
                            if args.trail_fade_mode == "immediate":
                                point_history[i].clear()
                            elif args.trail_fade_mode == "gradual":
                                if len(point_history[i]) > 1:
                                    for j in range(1, len(point_history[i])):
                                        pt1 = point_history[i][j - 1]
                                        pt2 = point_history[i][j]
                                        alpha = (j / len(point_history[i])) * 0.3
                                        color = apply_transparency(colors[i], alpha)
                                        cv2.line(frame, pt1, pt2, color, 2)

                                if len(point_history[i]) > 0:
                                    point_history[i].pop(0)

                # Convert back to RGB
                if C == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        elif mode == "overview":
            # Update point history for overview mode BEFORE creating the frame
            if (
                not args.no_trails
                and point_tracks is not None
                and point_visibility is not None
                and t < point_tracks.shape[0]
            ):
                current_tracks = point_tracks[t]
                current_visibility = point_visibility[t]

                for i in range(current_tracks.shape[0]):
                    if current_visibility[i, 0] > 0.5:
                        x, y = current_tracks[i, 0].item(), current_tracks[i, 1].item()
                        x = max(0, min(W - 1, int(x)))
                        y = max(0, min(H - 1, int(y)))

                        point_history[i].append((x, y))
                        if len(point_history[i]) > args.trail_length:
                            point_history[i].pop(0)
                    else:
                        if args.trail_fade_mode == "immediate":
                            point_history[i].clear()
                        elif args.trail_fade_mode == "gradual" and len(point_history[i]) > 0:
                            point_history[i].pop(0)

            frame = create_overview_frame(data, args, t, point_history, colors, depth_stats)
            if frame is None:
                frame = video[t].numpy().astype(np.uint8)

        else:
            # Default to video
            frame = video[t].numpy().astype(np.uint8)

        frames.append(frame)

        # Show progress
        if (t + 1) % 10 == 0 or t == T - 1:
            print(f"  Processed frame {t + 1}/{T}")

    return frames


def visualize_single_shard(args, key, shard):
    """Visualize a single key/shard combination"""

    print(f"Loading data for {key}/{shard}...")
    try:
        data = read_wods_shard(args.input, key, shard)
        metadata = read_wods_metadata(args.input, key)
    except Exception as e:
        print(f"Error reading wods shard {key}/{shard}: {e}")
        return False

    # Print available data types
    available_fields = list(data.keys())
    print(f"Available fields: {available_fields}")

    # Print metadata
    if metadata:
        caption = metadata.get("caption", "No caption")
        print(f"Caption: {caption}")

    # Get video info for timing
    video = data.get("video")
    if video is None:
        print(f"No video data found in shard {key}/{shard}")
        return False

    print(f"Video shape: {video.shape}")

    # Print shapes of other available data
    for field, tensor in data.items():
        if field != "video":
            if field == "mask" and isinstance(tensor, dict):
                print(f"{field} (dict): {{")
                for name, mask_tensor in tensor.items():
                    print(f"  {name}: {mask_tensor.shape}")
                print("}")
            elif hasattr(tensor, "shape"):
                print(f"{field} shape: {tensor.shape}")
            else:
                print(f"{field}: {type(tensor)}")

    T, H, W, C = video.shape

    # Determine output FPS
    timestamps = data.get("timestamps")
    original_fps = infer_fps_from_timestamps(timestamps) if timestamps is not None else None

    if original_fps is not None:
        output_fps = original_fps
        print(f"Inferred original FPS: {original_fps:.2f}")
    else:
        output_fps = args.fps
        print(f"Using default FPS: {args.fps} (timestamps not available or insufficient)")

    # Set up output path
    output_dir = Path(args.output)

    key_dir = output_dir / key
    output_path = key_dir / f"{shard}_{args.mode}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {args.mode} visualization to {output_path}")

    # Generate frames based on mode
    try:
        frames = visualize_single_mode(data, args, args.mode)

        if not frames:
            print(f"No frames generated for {args.mode} mode")
            return False

        # Write video
        print(f"Writing {len(frames)} frames to video at {output_fps:.2f} FPS...")
        iio.imwrite(str(output_path), frames, fps=output_fps)

        print(f"Visualization saved to {output_path}")
        return True

    except Exception as e:
        print(f"Error creating visualization: {e}")
        traceback.print_exc()
        return False


def visualize_wods(args):
    """Main visualization function"""

    # If both key and shard are provided, visualize single shard
    if args.key and args.shard:
        visualize_single_shard(args, args.key, args.shard)
        return

    # If only key is provided, visualize all shards for that key
    if args.key and not args.shard:
        try:
            shards = list(list_wods_shards(args.input, args.key))
            if not shards:
                print(f"No shards found for key '{args.key}'")
                return

            print(f"Visualizing all {len(shards)} shards for key '{args.key}' in {args.mode} mode:")
            for shard in shards:
                print(f"  {shard}")

            success_count = 0
            for shard in shards:
                if visualize_single_shard(args, args.key, shard):
                    success_count += 1

            print(f"Successfully visualized {success_count}/{len(shards)} shards for key '{args.key}'")
        except Exception as e:
            print(f"Error processing key '{args.key}': {e}")
        return

    # If neither key nor shard is provided, visualize all keys and their shards
    if not args.key and not args.shard:
        try:
            keys = list(list_wods_keys(args.input))
            if not keys:
                print("No keys found in dataset")
                return

            print(f"Visualizing all keys and shards in dataset in {args.mode} mode:")
            for key in keys[:10]:  # Show first 10 keys
                print(f"  {key}")
            if len(keys) > 10:
                print(f"  ... and {len(keys) - 10} more")

            total_success = 0
            total_shards = 0

            for key in keys:
                try:
                    shards = list(list_wods_shards(args.input, key))
                    total_shards += len(shards)

                    for shard in shards:
                        if visualize_single_shard(args, key, shard):
                            total_success += 1

                except Exception as e:
                    print(f"Error processing key '{key}': {e}")

            print(f"Successfully visualized {total_success}/{total_shards} total shards")
        except Exception as e:
            print(f"Error listing keys: {e}")
        return


def main():
    args = parse_args()
    visualize_wods(args)


if __name__ == "__main__":
    main()
