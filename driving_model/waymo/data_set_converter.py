"""
Waymo E2E Dataset -> HDF5 Converter
====================================
Converts Waymo End-to-End driving .tfrecord files to a single HDF5 file
that can be loaded directly with PyTorch — no TensorFlow required.

Usage:
    python convert_waymo_to_hdf5.py \
        --input "path/to/waymo_data/*.tfrecord" \
        --output "waymo_e2e.h5" \
        --split training

Requirements:
    pip install h5py numpy opencv-python protobuf
    pip install waymo-open-dataset-tf-2-12-0 --no-deps  (for protos only)

HDF5 Structure:
    /frames/
        {i}/
            images/
                left     (H, W, 3) uint8  JPEG-decoded
                center   (H, W, 3) uint8
                right    (H, W, 3) uint8
            waypoints_2d   (N, 2)  float32  [pos_x, pos_y]
            waypoints_3d   (N, 3)  float32  [pos_x, pos_y, pos_z]
            pose           (4, 4)  float32  vehicle->world transform
            calibration/
                left/
                    intrinsic    (9,)   float32
                    extrinsic    (4,4)  float32
                    width        scalar int
                    height       scalar int
                center/  (same)
                right/   (same)
    /metadata/
        num_frames   scalar
        split        string
        source_files list of strings
"""

import argparse
import glob
import os
import struct
import sys
import traceback

import cv2
import h5py
import numpy as np
from PIL import Image
import io

# ---------------------------------------------------------------------------
# Proto imports — only the proto parsing, no TF runtime needed
# ---------------------------------------------------------------------------
try:
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
    from waymo_open_dataset import dataset_pb2 as open_dataset
except ImportError:
    print("ERROR: Could not import waymo_open_dataset protos.")
    print("Install with: pip install waymo-open-dataset-tf-2-12-0 --no-deps")
    sys.exit(1)

# Camera name constants (from Waymo dataset_pb2)
CAMERA_NAMES = {
    1: "center",
    2: "left",
    3: "right",
    4: "side_left",
    5: "side_right",
}

# The three front cameras we care about, in left->center->right order
FRONT_CAMERAS = [2, 1, 3]
FRONT_CAMERA_NAMES = ["left", "center", "right"]


# ---------------------------------------------------------------------------
# TFRecord reader — pure Python, no TensorFlow
# ---------------------------------------------------------------------------

def _masked_crc32c(data: bytes) -> int:
    """Compute masked CRC32C as used in TFRecord format."""
    import struct
    try:
        import crcmod
        crc_fn = crcmod.predefined.mkCrcFun('crc-32c')
        crc = crc_fn(data)
    except ImportError:
        # Fallback: skip CRC validation if crcmod not available
        return 0
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff


def read_tfrecord_file(path: str):
    """
    Generator that yields raw serialized proto bytes from a TFRecord file.
    Implements the TFRecord format: length (8B) + masked_crc (4B) + data + masked_crc (4B)
    """
    with open(path, 'rb') as f:
        while True:
            # Read the length of the next record
            len_bytes = f.read(8)
            if len_bytes == b'':
                break  # EOF
            if len(len_bytes) < 8:
                print(f"  WARNING: Truncated length field in {path}, stopping.")
                break

            length = struct.unpack('<Q', len_bytes)[0]
            f.read(4)  # masked CRC of length (skip validation for speed)

            data = f.read(length)
            if len(data) < length:
                print(f"  WARNING: Truncated data in {path}, stopping.")
                break

            f.read(4)  # masked CRC of data (skip validation for speed)
            yield data


# ---------------------------------------------------------------------------
# Frame parsing helpers
# ---------------------------------------------------------------------------

def decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes to (H, W, 3) uint8 numpy array (RGB)."""
    try:
        # Load the bytes into an in-memory file stream
        stream = io.BytesIO(raw_bytes)
        
        # Open the image with Pillow and ensure it's RGB
        img = Image.open(stream).convert("RGB")
        
        # Convert directly to a NumPy array
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"  WARNING: Failed to decode image: {e}")
        return None


def parse_calibration(calib) -> dict:
    """Extract intrinsic, extrinsic, width, height from a CameraCalibration proto."""
    return {
        "intrinsic": np.array(list(calib.intrinsic), dtype=np.float32),       # (9,)
        "extrinsic": np.array(list(calib.extrinsic.transform),
                               dtype=np.float32).reshape(4, 4),                # (4,4)
        "width":  calib.width,
        "height": calib.height,
    }


def parse_frame(raw_bytes: bytes) -> dict | None:
    """
    Parse a single serialized E2EDFrame proto into a flat dict of numpy arrays.
    Returns None if the frame is missing required data.
    """
    data = wod_e2ed_pb2.E2EDFrame()
    try:
        data.ParseFromString(raw_bytes)
    except Exception as e:
        print(f"  WARNING: Failed to parse frame proto: {e}")
        return None

    # ---- Future waypoints ----
    try:
        waypoints_2d = np.stack([
            np.array(data.future_states.pos_x, dtype=np.float32),
            np.array(data.future_states.pos_y, dtype=np.float32),
        ], axis=1)  # (N, 2)

        waypoints_3d = np.stack([
            np.array(data.future_states.pos_x, dtype=np.float32),
            np.array(data.future_states.pos_y, dtype=np.float32),
            np.array(data.future_states.pos_z, dtype=np.float32),
        ], axis=1)  # (N, 3)
    except Exception as e:
        print(f"  WARNING: Missing waypoint data: {e}")
        return None

    # ---- Pose (from first image, vehicle->world) ----
    pose = None
    if data.frame.images:
        try:
            pose = np.array(data.frame.images[0].pose.transform,
                            dtype=np.float32).reshape(4, 4)
        except Exception:
            pose = np.eye(4, dtype=np.float32)
    else:
        pose = np.eye(4, dtype=np.float32)

    # ---- Images and calibrations (front 3 cameras) ----
    # Build lookup dicts: camera_name_id -> image proto, calibration proto
    image_by_name = {img.name: img for img in data.frame.images}
    calib_by_name = {c.name: c for c in data.frame.context.camera_calibrations}

    images = {}
    calibrations = {}

    for cam_id, cam_label in zip(FRONT_CAMERAS, FRONT_CAMERA_NAMES):
        if cam_id not in image_by_name:
            print(f"  WARNING: Camera {cam_label} (id={cam_id}) not found in frame.")
            return None

        img_proto = image_by_name[cam_id]
        decoded = decode_image(bytes(img_proto.image))
        if decoded is None:
            print(f"  WARNING: Failed to decode image for camera {cam_label}.")
            return None
        images[cam_label] = decoded

        if cam_id in calib_by_name:
            calibrations[cam_label] = parse_calibration(calib_by_name[cam_id])
        else:
            # Fallback empty calibration
            calibrations[cam_label] = {
                "intrinsic": np.zeros(9, dtype=np.float32),
                "extrinsic": np.eye(4, dtype=np.float32),
                "width": decoded.shape[1],
                "height": decoded.shape[0],
            }

    return {
        "images": images,               # dict: label -> (H,W,3) uint8
        "waypoints_2d": waypoints_2d,   # (N,2) float32
        "waypoints_3d": waypoints_3d,   # (N,3) float32
        "pose": pose,                   # (4,4) float32
        "calibrations": calibrations,   # dict: label -> {intrinsic,extrinsic,width,height}
    }


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_frame_to_hdf5(frames_group: h5py.Group, frame_idx: int, frame: dict):
    """Write a single parsed frame into the HDF5 frames group."""
    fg = frames_group.create_group(str(frame_idx))

    # Images
    img_group = fg.create_group("images")
    for cam_label, img_arr in frame["images"].items():
        img_group.create_dataset(cam_label, data=img_arr, compression="lzf")

    # Waypoints
    fg.create_dataset("waypoints_2d", data=frame["waypoints_2d"])
    fg.create_dataset("waypoints_3d", data=frame["waypoints_3d"])

    # Pose
    fg.create_dataset("pose", data=frame["pose"])

    # Calibrations
    calib_group = fg.create_group("calibration")
    for cam_label, calib in frame["calibrations"].items():
        cg = calib_group.create_group(cam_label)
        cg.create_dataset("intrinsic", data=calib["intrinsic"])
        cg.create_dataset("extrinsic", data=calib["extrinsic"])
        cg.attrs["width"] = calib["width"]
        cg.attrs["height"] = calib["height"]


# ---------------------------------------------------------------------------
# Main conversion loop
# ---------------------------------------------------------------------------

def convert(input_pattern: str, output_path: str, split: str, max_frames: int = None):
    files = sorted(glob.glob(input_pattern))
    if not files:
        print(f"ERROR: No files matched pattern: {input_pattern}")
        sys.exit(1)

    print(f"Found {len(files)} tfrecord file(s).")
    print(f"Output: {output_path}")

    frame_idx = 0
    skipped = 0

    with h5py.File(output_path, 'w') as hf:
        frames_group = hf.create_group("frames")
        meta_group = hf.create_group("metadata")
        meta_group.attrs["split"] = split
        meta_group.attrs["source_files"] = [os.path.basename(f) for f in files]

        for file_path in files:
            print(f"\nProcessing: {os.path.basename(file_path)}")
            file_frames = 0

            for raw_bytes in read_tfrecord_file(file_path):
                if max_frames is not None and frame_idx >= max_frames:
                    print(f"Reached max_frames={max_frames}, stopping.")
                    break

                frame = parse_frame(raw_bytes)
                if frame is None:
                    skipped += 1
                    continue

                try:
                    write_frame_to_hdf5(frames_group, frame_idx, frame)
                    frame_idx += 1
                    file_frames += 1

                    if frame_idx % 50 == 0:
                        print(f"  Written {frame_idx} frames so far...")

                except Exception as e:
                    print(f"  WARNING: Failed to write frame {frame_idx}: {e}")
                    traceback.print_exc()
                    skipped += 1

            print(f"  Done — {file_frames} frames from this file.")

            if max_frames is not None and frame_idx >= max_frames:
                break

        meta_group.attrs["num_frames"] = frame_idx
        print(f"\nConversion complete.")
        print(f"  Total frames written : {frame_idx}")
        print(f"  Frames skipped       : {skipped}")
        print(f"  Output file          : {output_path}")


# ---------------------------------------------------------------------------
# PyTorch Dataset (ready to use after conversion)
# ---------------------------------------------------------------------------

PYTORCH_DATASET_CODE = '''
# ============================================================
# WaymoHDF5Dataset — drop this into your training script
# ============================================================
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class WaymoHDF5Dataset(Dataset):
    """
    Loads pre-converted Waymo E2E data from an HDF5 file.

    Args:
        h5_path:   Path to the .h5 file produced by convert_waymo_to_hdf5.py
        cameras:   Which cameras to return. Default: ["left", "center", "right"]
        transform: Optional torchvision transform applied to each image.
        concat_images: If True, concatenate images along width axis before transform.
    """
    def __init__(self, h5_path, cameras=None, transform=None, concat_images=True):
        self.h5_path = h5_path
        self.cameras = cameras or ["left", "center", "right"]
        self.transform = transform
        self.concat_images = concat_images

        with h5py.File(h5_path, 'r') as hf:
            self.num_frames = hf["metadata"].attrs["num_frames"]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as hf:
            fg = hf[f"frames/{idx}"]

            # Images
            imgs = [fg[f"images/{cam}"][:] for cam in self.cameras]

            if self.concat_images:
                image = np.concatenate(imgs, axis=1)  # (H, W*3, 3)
                if self.transform:
                    image = self.transform(image)
            else:
                if self.transform:
                    imgs = [self.transform(im) for im in imgs]
                image = imgs  # list of tensors

            # Waypoints
            waypoints_2d = torch.from_numpy(fg["waypoints_2d"][:])
            waypoints_3d = torch.from_numpy(fg["waypoints_3d"][:])

            # Pose
            pose = torch.from_numpy(fg["pose"][:])

            # Calibration (all cameras)
            calibration = {}
            for cam in self.cameras:
                cg = fg[f"calibration/{cam}"]
                calibration[cam] = {
                    "intrinsic": torch.from_numpy(cg["intrinsic"][:]),
                    "extrinsic": torch.from_numpy(cg["extrinsic"][:]),
                    "width":  cg.attrs["width"],
                    "height": cg.attrs["height"],
                }

        return {
            "image": image,
            "waypoints_2d": waypoints_2d,
            "waypoints_3d": waypoints_3d,
            "pose": pose,
            "calibration": calibration,
        }


# Example usage:
# from torchvision import transforms
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
# ])
# dataset = WaymoHDF5Dataset("waymo_e2e_training.h5", transform=transform)
# loader  = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
'''


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Waymo E2E tfrecord files to HDF5 for PyTorch."
    )
    parser.add_argument(
        "--input", required=True,
        help='Glob pattern for input tfrecord files. E.g. "data/training_*.tfrecord"'
    )
    parser.add_argument(
        "--output", required=True,
        help="Output HDF5 file path. E.g. waymo_e2e_training.h5"
    )
    parser.add_argument(
        "--split", default="training",
        choices=["training", "validation", "testing"],
        help="Dataset split label stored in metadata."
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Optional: stop after this many frames (useful for testing)."
    )
    parser.add_argument(
        "--print_dataset_code", action="store_true",
        help="Print the WaymoHDF5Dataset class to use in your training script."
    )
    args = parser.parse_args()

    if args.print_dataset_code:
        print(PYTORCH_DATASET_CODE)
        sys.exit(0)

    convert(
        input_pattern=args.input,
        output_path=args.output,
        split=args.split,
        max_frames=args.max_frames,
    )