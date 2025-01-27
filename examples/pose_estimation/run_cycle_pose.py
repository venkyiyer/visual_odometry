import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.append(str(src_path))

from pose_estimation.interface import PoseEstimator
import argparse
from pathlib import Path
import numpy as np
import time


def process_image_pair(estimator, frame1_path: str, frame2_path: str):
    """Process a single pair of images."""
    print(f"\nProcessing image pair:")
    print(f"Frame 1: {frame1_path}")
    print(f"Frame 2: {frame2_path}")

    relative_pose = estimator.estimate_relative_pose(frame1_path, frame2_path)
    print("\nEstimated relative pose (4x4 transformation matrix):")
    print(relative_pose)

    return relative_pose


def process_single_sequence(estimator, sequence_path: str, output_path: str):
    """Process a single sequence of images."""
    sequence_dir = Path(sequence_path)
    print(f"\nProcessing sequence from: {sequence_dir}")

    # Get all images in the sequence
    image_paths = sorted(list(sequence_dir.glob("*.jpg")))
    if not image_paths:
        image_paths = sorted(list(sequence_dir.glob("*.png")))

    if not image_paths:
        print("No images found in sequence directory!")
        return None

    print(f"Found {len(image_paths)} images")

    # Process the sequence
    start_time = time.time()
    poses = estimator.process_sequence(image_paths, output_path)
    processing_time = time.time() - start_time

    print(f"\nProcessing complete:")
    print(f"- Time taken: {processing_time:.2f} seconds")
    print(f"- Average time per frame: {processing_time / len(image_paths):.3f} seconds")
    print(f"- Poses saved to: {output_path}")

    return poses


def process_dataset(estimator, dataset_path: str, output_dir: str):
    """Process an entire dataset with multiple sequences."""
    dataset_dir = Path(dataset_path)
    print(f"\nProcessing dataset from: {dataset_dir}")

    # Process all sequences
    start_time = time.time()
    results = estimator.process_dataset(dataset_path, output_dir)
    processing_time = time.time() - start_time

    print(f"\nDataset processing complete:")
    print(f"- Processed {len(results)} sequences")
    print(f"- Total time taken: {processing_time:.2f} seconds")
    print(f"- Results saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Deep Learning Pose Estimation Example")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model weights')
    parser.add_argument('--mode', type=str, choices=['pair', 'sequence', 'dataset'],
                        required=True, help='Processing mode')
    parser.add_argument('--input', type=str, required=True,
                        help='Input path (varies based on mode)')
    parser.add_argument('--input2', type=str,
                        help='Second image path (only for pair mode)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for results')

    args = parser.parse_args()

    # Initialize the pose estimator
    print(f"Initializing pose estimator with model: {args.model_path}")
    estimator = PoseEstimator(args.model_path)

    # Process based on mode
    if args.mode == 'pair':
        if not args.input2:
            raise ValueError("Second image path (--input2) required for pair mode")
        process_image_pair(estimator, args.input, args.input2)

    elif args.mode == 'sequence':
        process_single_sequence(estimator, args.input, args.output)

    elif args.mode == 'dataset':
        process_dataset(estimator, args.input, args.output)


if __name__ == "__main__":
    main()

