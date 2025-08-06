from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from .Architecture.cycle_vo import CycleVO


class PoseEstimator:
    """
    Interface for estimating relative poses between consecutive frames
    using the CycleVO architecture.
    """

    def __init__(self, model_path: str, input_shape: tuple = (6, 256, 256)):
        """
        Initialize the pose estimator.

        Args:
            model_path: Path to the trained weights
            input_shape: Input shape for the model (channels, height, width)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_shape = input_shape
        self.model = self._load_model(model_path)
        self.transform = self._setup_transforms()

    def _load_model(self, model_path: str) -> nn.Module:
        """Load the CycleVO model with pre-trained weights."""
        # Initialize model
        model = CycleVO(
            device=self.device,
            input_shape=self.input_shape
        ).to(self.device)

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    def _setup_transforms(self) -> transforms.Compose:
        """Setup image transformations matching the model's expected input."""
        print("Reaching setup transforms -->")
        return transforms.Compose([
            transforms.Resize((self.input_shape[1], self.input_shape[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image."""
        image = Image.open(image_path).convert('RGB')
        print("Image path ->", type(image))
        return self.transform(image).unsqueeze(0)

    def estimate_relative_pose(self, frame1_path: str, frame2_path: str) -> np.ndarray:
        """
        Estimate the relative pose between two consecutive frames.

        Args:
            frame1_path: Path to the first frame
            frame2_path: Path to the second frame

        Returns:
            4x4 transformation matrix as numpy array
        """
        # Load and preprocess images
        frame1 = self._load_and_preprocess_image(frame1_path).to(self.device)
        frame2 = self._load_and_preprocess_image(frame2_path).to(self.device)

        # Concatenate frames
        frames = torch.cat([frame1, frame2], dim=1)

        # Get pose prediction
        with torch.no_grad():
            relative_pose = self.model(frames, mode='pose')

        return relative_pose.squeeze().cpu().numpy()

    def process_sequence(self,
                         image_paths: List[str],
                         output_path: Optional[str] = None, progress_callback=None) -> List[np.ndarray]:
        """
        Process a sequence of images and compute relative poses.

        Args:
            image_paths: List of paths to images in sequence order
            output_path: Optional path to save the poses

        Returns:
            List of 4x4 transformation matrices
        """
        poses = [np.eye(4)]  # Start with identity matrix
        current_pose = np.eye(4)

        print("Processing image sequence...")
        total = len(image_paths) - 1
        for i in tqdm(range(len(image_paths) - 1)):
            # Get relative pose between consecutive frames
            relative_pose = self.estimate_relative_pose(
                image_paths[i],
                image_paths[i + 1]
            )

            # Update current pose
            current_pose = current_pose @ relative_pose
            poses.append(current_pose.copy())

            if progress_callback:
                progress_callback(i + 1, total)
        # Save poses if output path is provided
        if output_path:
            self.save_poses(poses, output_path)

        return poses

    def process_dataset(self,
                        dataset_path: str,
                        output_dir: str) -> Dict[str, List[np.ndarray]]:
        """
        Process all sequences in a dataset.

        Args:
            dataset_path: Path to dataset root directory
            output_dir: Directory to save pose files

        Returns:
            Dictionary mapping sequence names to their poses
        """
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Process each sequence in the dataset
        for sequence_dir in dataset_path.iterdir():
            if not sequence_dir.is_dir():
                continue

            print(f"\nProcessing sequence: {sequence_dir.name}")

            # Get sorted list of image paths
            image_paths = sorted(list(sequence_dir.glob("*.jpg")))
            if not image_paths:
                image_paths = sorted(list(sequence_dir.glob("*.png")))

            if not image_paths:
                print(f"No images found in {sequence_dir}")
                continue

            # Process sequence
            output_path = output_dir / f"{sequence_dir.name}_poses.txt"
            poses = self.process_sequence(image_paths, str(output_path))
            results[sequence_dir.name] = poses

        return results

    @staticmethod
    def save_poses(poses: List[np.ndarray], output_path: str):
        """
        Save poses in KITTI format.

        Args:
            poses: List of 4x4 transformation matrices
            output_path: Path to save the pose file
        """
        with open(output_path, 'w') as f:
            for pose in poses:
                # Convert to KITTI format (first 12 elements of flattened matrix)
                line = " ".join(map(str, pose.flatten()[:12]))
                f.write(f"{line}\n")


# Example usage:
if __name__ == "__main__":
    # Initialize pose estimator
    estimator = PoseEstimator(
        model_path="path/to/model.pth",
        input_shape=(6, 256, 256)  # Default shape for CycleVO
    )

    # Process a dataset
    dataset_poses = estimator.process_dataset(
        dataset_path="path/to/dataset",
        output_dir="path/to/output"
    )


