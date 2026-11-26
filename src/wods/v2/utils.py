import os
import tempfile
import time
from contextlib import contextmanager

import fsspec
import imageio.v3 as iio
import Imath
import numpy as np
import OpenEXR
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def image_u8_to_float(image):
    return image / 255.0 * 2 - 1


def image_float_to_u8(image):
    return ((image + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)


def write_video_yuv(uri, video, crf=23, qp=None, **kwargs):
    output_params = []
    if qp is not None:
        output_params += ["-qp", str(qp)]
    elif crf is not None:
        output_params += ["-crf", str(crf)]

    bytes = iio.imwrite(
        "<bytes>",
        video,
        extension=".mp4",
        plugin="FFMPEG",
        macro_block_size=1,
        output_params=output_params,
        **kwargs,
    )
    if isinstance(uri, str):
        with fsspec.open(uri, "wb") as f:
            f.write(bytes)
    else:
        uri.write(bytes)


def write_video_rgb(uri, video, crf=23, qp=None, **kwargs):
    output_params = []
    if qp is not None:
        output_params += ["-qp", str(qp)]
    elif crf is not None:
        output_params += ["-crf", str(crf)]

    bytes = iio.imwrite(
        "<bytes>",
        video,
        extension=".mp4",
        plugin="FFMPEG",
        codec="libx264rgb",
        pixelformat="rgb24",
        macro_block_size=1,
        output_params=output_params,
        **kwargs,
    )
    if isinstance(uri, str):
        with fsspec.open(uri, "wb") as f:
            f.write(bytes)
    else:
        uri.write(bytes)


def grid_arange(*args, **kwargs):
    seqs = []
    for i in range(0, len(args), 3):
        seqs.append(torch.arange(args[i], args[i + 1], args[i + 2], **kwargs))
    return torch.stack(torch.meshgrid(*seqs, indexing="ij"), dim=-1)


@contextmanager
def fsspec_open_with_retry(path, mode="r", sleep_time=1.0):
    """
    Open a file using fsspec with retry logic for any exception.
    Retries indefinitely until successful.

    Args:
        path: File path to open
        mode: File mode (e.g., "r", "rb")
        sleep_time: Time to sleep between retries in seconds
    """
    attempt = 0
    while True:
        try:
            with fsspec.open(path, mode) as f:
                yield f
            return
        except Exception as e:
            attempt += 1
            print(f"fsspec.open failed for {path} (attempt {attempt}): {e}")
            time.sleep(sleep_time)


def fsspec_glob_with_retry(path, pattern, sleep_time=1.0):
    """
    Perform fsspec glob operation with retry logic for any exception.
    Retries indefinitely until successful.

    Args:
        path: Base path for filesystem
        pattern: Glob pattern to match
        sleep_time: Time to sleep between retries in seconds
    """
    attempt = 0
    while True:
        try:
            fs = fsspec.open(path).fs
            return fs.glob(pattern)
        except Exception as e:
            attempt += 1
            print(f"fsspec.glob failed for {pattern} (attempt {attempt}): {e}")
            time.sleep(sleep_time)


def resize_camera_intrinsics(K, original_size, new_size):
    orig_h, orig_w = original_size
    new_h, new_w = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    K_new = K.clone()
    K_new[..., 0, 0] *= scale_x  # fx
    K_new[..., 1, 1] *= scale_y  # fy
    K_new[..., 0, 2] *= scale_x  # cx
    K_new[..., 1, 2] *= scale_y  # cy

    return K_new


def center_crop_camera_intrinsics(K, original_size, new_size):
    orig_h, orig_w = original_size
    new_h, new_w = new_size

    offset_x = (orig_w - new_w) / 2
    offset_y = (orig_h - new_h) / 2

    K_new = K.clone()
    K_new[..., 0, 2] -= offset_x  # cx
    K_new[..., 1, 2] -= offset_y  # cy

    return K_new


def crop_resize(x, size, intrinsics, interpolation="bilinear"):
    # x: ...HWC
    # size: (int, int)
    # intrinsics: ...(3)(3)
    # interpolation: str
    shape = x.shape
    h, w = x.shape[-3:-1]

    x = x.permute(*range(x.dim() - 3), -1, -3, -2)  # ...CHW
    x = x.flatten(0, -4)  # (...)CHW

    x_ratio = h / w
    size_ratio = size[0] / size[1]

    if x_ratio > size_ratio:
        # Tall
        new_h = int(h / w * size[1])
        new_w = size[1]
        x = F.interpolate(x, size=(new_h, new_w), mode=interpolation)
        if intrinsics is not None:
            intrinsics = resize_camera_intrinsics(intrinsics, (h, w), (new_h, new_w))
    else:
        # Wide
        new_h = size[0]
        new_w = int(w / h * size[0])
        x = F.interpolate(x, size=(new_h, new_w), mode=interpolation)
        if intrinsics is not None:
            intrinsics = resize_camera_intrinsics(intrinsics, (h, w), (new_h, new_w))

    x = TF.center_crop(x, size)
    if intrinsics is not None:
        intrinsics = center_crop_camera_intrinsics(intrinsics, (new_h, new_w), size)

    x = x.unflatten(0, shape[:-3])  # ...CHW
    x = x.permute(*range(x.dim() - 3), -2, -1, -3)  # ...HWC
    return x, intrinsics


def get_rotation_matrix(rotation):
    # rotation: ...3
    c0 = rotation[..., 0].cos()
    s0 = rotation[..., 0].sin()
    c1 = rotation[..., 1].cos()
    s1 = rotation[..., 1].sin()
    c2 = rotation[..., 2].cos()
    s2 = rotation[..., 2].sin()

    z = torch.zeros_like(c0)
    o = torch.ones_like(c0)

    rx = torch.stack(
        [
            torch.stack([o, z, z], dim=-1),
            torch.stack([z, c0, s0], dim=-1),
            torch.stack([z, -s0, c0], dim=-1),
        ],
        dim=-2,
    )
    ry = torch.stack(
        [
            torch.stack([c1, z, s1], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s1, z, c1], dim=-1),
        ],
        dim=-2,
    )
    rz = torch.stack(
        [
            torch.stack([c2, s2, z], dim=-1),
            torch.stack([-s2, c2, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ],
        dim=-2,
    )

    return rz @ ry @ rx


def camera_focal(image_width, fov):
    return (image_width / 2) / torch.tan(fov / 2)  # ...


def camera_fov(image_width, focal):
    return 2 * torch.atan(image_width / 2 / focal)  # ...


def camera_intrinsic(image_size, fov):
    focal = camera_focal(image_size[1], fov)  # ...
    z = torch.zeros_like(focal)
    o = torch.ones_like(focal)
    intrinsic = torch.stack(
        [
            torch.stack([focal, z, torch.full_like(z, image_size[1] / 2)], dim=-1),
            torch.stack([z, focal, torch.full_like(z, image_size[0] / 2)], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ],
        dim=-2,
    )  # ...(3)(3)
    return intrinsic


def camera_intrinsic_with_zero_center(image_size, fov):
    focal = camera_focal(image_size[1], fov)  # ...
    z = torch.zeros_like(focal)
    o = torch.ones_like(focal)
    intrinsic = torch.stack(
        [
            torch.stack([focal, z, z], dim=-1),
            torch.stack([z, focal, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ],
        dim=-2,
    )  # ...(3)(3)
    return intrinsic


def normalize_traj_c2w(timesteps, c2ws, timesteps_base, c2ws_base):
    # timesteps: ...T
    # c2ws: ...T(3)(4)
    # timesteps_base: ...T
    # c2ws_base: ...T(3)(4)

    # Time
    timesteps = timesteps - timesteps_base[..., :1]  # ...T

    positions = c2ws[..., 3]  # ...T3
    rotation_matrix = c2ws[..., :3]  # ...T(3)(3)

    positions_base = c2ws_base[..., 3]  # ...T3
    rotation_base_matrix = c2ws_base[..., :3]  # ...T(3)(3)

    # Rotation
    rotation_base_matrix_inv = rotation_base_matrix[..., :1, :, :].inverse()  # ...T(3)(3)
    positions_base = torch.einsum("...ij, ...j -> ...i", rotation_base_matrix_inv, positions_base)  # ...T3
    positions = torch.einsum("...ij, ...j -> ...i", rotation_base_matrix_inv, positions)  # ...T3
    rotation_matrix = torch.einsum("...ij, ...jk -> ...ik", rotation_base_matrix_inv, rotation_matrix)  # ...T(3)(3)

    # Position
    positions = positions - positions_base[..., :1, :]  # ...T3

    c2ws = torch.cat([rotation_matrix, positions[..., None]], dim=-1)  # ...T(3)(4)
    return timesteps, c2ws


def get_distributed_slice(total_length, rank=None, world_size=None):
    """
    Calculate start and end indices for distributed processing based on RANK and WORLD_SIZE.

    Args:
        total_length: Total number of items to process
        rank: Process rank (defaults to RANK environment variable)
        world_size: Total number of processes (defaults to WORLD_SIZE environment variable)

    Returns:
        tuple: (start_index, end_index) for this process
    """
    if rank is None:
        rank = int(os.getenv("RANK", "0"))
    if world_size is None:
        world_size = int(os.getenv("WORLD_SIZE", "1"))

    # Calculate items per process
    items_per_process = total_length // world_size
    remainder = total_length % world_size

    # Calculate start index for this rank
    start_idx = rank * items_per_process
    if rank < remainder:
        start_idx += rank
    else:
        start_idx += remainder

    # Calculate end index for this rank
    if rank < remainder:
        end_idx = start_idx + items_per_process + 1
    else:
        end_idx = start_idx + items_per_process

    return start_idx, end_idx


def write_depth_exr(file_obj, depth_tensor):
    """Write depth data to EXR format. Supports only THW shapes and file-like objects."""
    depth_array = depth_tensor.detach().cpu().numpy().astype(np.float32)

    if depth_array.ndim == 2:
        depth_array = depth_array[None, ...]  # H,W -> 1,H,W

    T, H, W = depth_array.shape

    # Use OpenEXR for all cases
    channels = {}
    header_channels = {}

    for i in range(T):
        channel_name = f"Z{i:03d}" if i > 0 else "Z"
        channels[channel_name] = depth_array[i].tobytes()
        header_channels[channel_name] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))

    header = OpenEXR.Header(W, H)
    header["channels"] = header_channels

    with tempfile.NamedTemporaryFile(suffix=".exr") as temp_file:
        exr_file = OpenEXR.OutputFile(temp_file.name, header)
        exr_file.writePixels(channels)
        exr_file.close()

        temp_file.seek(0)
        file_obj.write(temp_file.read())


def read_depth_exr(file_obj):
    """Read depth data from EXR format. Returns tensor with THW shape."""
    data = file_obj.read()

    with tempfile.NamedTemporaryFile(suffix=".exr") as temp_file:
        temp_file.write(data)
        temp_file.flush()

        exr_file = OpenEXR.InputFile(temp_file.name)
        header = exr_file.header()
        dw = header["dataWindow"]
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1

        channels = header["channels"]
        depth_channels = sorted([ch for ch in channels.keys() if ch.startswith("Z")])

        depth_maps = []
        for channel_name in depth_channels:
            channel_data = exr_file.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))
            depth_array = np.frombuffer(channel_data, dtype=np.float32).reshape((H, W))
            depth_maps.append(depth_array)

        exr_file.close()

        return torch.from_numpy(np.stack(depth_maps, axis=0).copy())
