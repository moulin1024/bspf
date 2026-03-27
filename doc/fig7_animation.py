#!/usr/bin/env python3
"""
Create MP4 animation from PNG frames in image_out directory.
"""

import os
import glob
import re
from pathlib import Path

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
    # Check if ffmpeg plugin is available
    try:
        import imageio_ffmpeg
        HAS_FFMPEG = True
        # Register ffmpeg plugin
        imageio.plugins.ffmpeg.download()
    except ImportError:
        HAS_FFMPEG = False
        print("Warning: imageio-ffmpeg not found. Install with: pip install imageio-ffmpeg")
    except Exception:
        # Plugin might already be registered
        HAS_FFMPEG = True
except ImportError:
    HAS_IMAGEIO = False
    HAS_FFMPEG = False
    print("Warning: imageio not found. Install with: pip install imageio imageio-ffmpeg")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def create_animation_imageio(input_dir="image_out", output_file="figs/fig7_animation.mp4", 
                            pattern="Shallow_water_2D_bfpsm_filtered_*.png", fps=10):
    """
    Create MP4 animation using imageio.
    
    Parameters
    ----------
    input_dir : str
        Directory containing PNG frames
    output_file : str
        Output MP4 file path
    pattern : str
        Glob pattern to match frame files
    fps : float
        Frames per second for the animation
    """
    if not HAS_IMAGEIO:
        raise ImportError("imageio is required. Install with: pip install imageio imageio-ffmpeg")
    
    # Get all matching files
    input_path = Path(input_dir)
    # Extract frame number from filename (match the last sequence of digits)
    # Pattern matches underscore followed by digits at the end: _0000, _0001, etc.
    def get_frame_number(path):
        # Try to match underscore followed by digits at the end (e.g., _0000, _0001)
        match = re.search(r'_(\d+)$', path.stem)
        if match:
            return int(match.group(1))
        # Fallback: try to find any 4-digit number (for files like filtered_0000)
        match = re.search(r'(\d{4})', path.stem)
        if match:
            return int(match.group(1))
        # Last resort: find last sequence of digits
        matches = re.findall(r'\d+', path.stem)
        if matches:
            return int(matches[-1])
        return 0
    
    frame_files = sorted(input_path.glob(pattern), key=get_frame_number)
    
    if len(frame_files) == 0:
        raise ValueError(f"No frames found matching pattern '{pattern}' in '{input_dir}'")
    
    print(f"Found {len(frame_files)} frames")
    print(f"First frame: {frame_files[0].name}")
    print(f"Last frame: {frame_files[-1].name}")
    # Debug: show frame numbers to verify sorting
    if len(frame_files) <= 10:
        print("Frame order:", [get_frame_number(f) for f in frame_files])
    else:
        print(f"Frame order (first 5): {[get_frame_number(f) for f in frame_files[:5]]}")
        print(f"Frame order (last 5): {[get_frame_number(f) for f in frame_files[-5:]]}")
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read frames and create animation
    print(f"Creating animation: {output_file}")
    print(f"FPS: {fps}")
    
    # Use ffmpeg plugin explicitly for MP4
    if not HAS_FFMPEG:
        raise ImportError("imageio-ffmpeg is required for MP4 creation. Install with: pip install imageio-ffmpeg")
    
    # Ensure output file has .mp4 extension
    if not output_file.endswith('.mp4'):
        output_file = str(Path(output_file).with_suffix('.mp4'))
    
    # Use ffmpeg plugin explicitly - specify plugin by name
    # The 'ffmpeg' format should use the FFMPEG plugin
    writer = imageio.get_writer(
        output_file, 
        fps=fps, 
        format='ffmpeg',
        codec='libx264',
        quality=8,
        pixelformat='yuv420p'  # Ensure compatibility
    )
    
    try:
        for i, frame_file in enumerate(frame_files):
            if i % 5 == 0:  # Progress indicator
                print(f"  Processing frame {i+1}/{len(frame_files)}: {frame_file.name}")
            img = imageio.imread(frame_file)
            writer.append_data(img)
    finally:
        writer.close()
    
    print(f"Animation saved to: {output_file}")


def create_animation_opencv(input_dir="image_out", output_file="figs/fig7_animation.mp4",
                          pattern="Shallow_water_2D_bfpsm_filtered_*.png", fps=10):
    """
    Create MP4 animation using OpenCV.
    
    Parameters
    ----------
    input_dir : str
        Directory containing PNG frames
    output_file : str
        Output MP4 file path
    pattern : str
        Glob pattern to match frame files
    fps : float
        Frames per second for the animation
    """
    if not HAS_CV2:
        raise ImportError("opencv-python is required. Install with: pip install opencv-python")
    
    # Get all matching files
    input_path = Path(input_dir)
    # Extract frame number from filename (match the last sequence of digits)
    # Pattern matches underscore followed by digits at the end: _0000, _0001, etc.
    def get_frame_number(path):
        # Try to match underscore followed by digits at the end (e.g., _0000, _0001)
        match = re.search(r'_(\d+)$', path.stem)
        if match:
            return int(match.group(1))
        # Fallback: try to find any 4-digit number (for files like filtered_0000)
        match = re.search(r'(\d{4})', path.stem)
        if match:
            return int(match.group(1))
        # Last resort: find last sequence of digits
        matches = re.findall(r'\d+', path.stem)
        if matches:
            return int(matches[-1])
        return 0
    
    frame_files = sorted(input_path.glob(pattern), key=get_frame_number)
    
    if len(frame_files) == 0:
        raise ValueError(f"No frames found matching pattern '{pattern}' in '{input_dir}'")
    
    print(f"Found {len(frame_files)} frames")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_files[0]}")
    
    height, width, channels = first_frame.shape
    print(f"Frame dimensions: {width}x{height}")
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_file}")
    
    print(f"Creating animation: {output_file}")
    print(f"FPS: {fps}")
    
    # Write frames
    for i, frame_file in enumerate(frame_files):
        if i % 5 == 0:  # Progress indicator
            print(f"  Processing frame {i+1}/{len(frame_files)}: {frame_file.name}")
        img = cv2.imread(str(frame_file))
        if img is None:
            print(f"Warning: Could not read {frame_file.name}, skipping")
            continue
        out.write(img)
    
    out.release()
    print(f"Animation saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create MP4 animation from PNG frames")
    parser.add_argument("--input-dir", default="image_out", help="Input directory with PNG frames")
    parser.add_argument("--output", default="figs/fig7_animation.mp4", help="Output MP4 file")
    parser.add_argument("--pattern", default="Shallow_water_2D_bfpsm_filtered_*.png", 
                       help="Glob pattern for frame files")
    parser.add_argument("--fps", type=float, default=10, help="Frames per second")
    parser.add_argument("--method", choices=["imageio", "opencv", "auto"], default="auto",
                       help="Method to use (auto selects available)")
    
    args = parser.parse_args()
    
    # Auto-select method
    if args.method == "auto":
        if HAS_IMAGEIO:
            method = "imageio"
        elif HAS_CV2:
            method = "opencv"
        else:
            raise ImportError("Neither imageio nor opencv-python found. Install one of them.")
    else:
        method = args.method
    
    # Create animation
    if method == "imageio":
        create_animation_imageio(args.input_dir, args.output, args.pattern, args.fps)
    elif method == "opencv":
        create_animation_opencv(args.input_dir, args.output, args.pattern, args.fps)
