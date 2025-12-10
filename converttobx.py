import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class PixelwiseToBBoxConverter:
    """Convert pixel-wise segmentation masks to bounding box masks"""

    def __init__(self, dataset_dir, images_subdir='images', masks_subdir='masks'):
        """
        Initialize converter

        Args:
            dataset_dir: Root directory containing images and masks subdirectories
            images_subdir: Name of subdirectory containing images (default: 'images')
            masks_subdir: Name of subdirectory containing masks (default: 'masks')
        """
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / images_subdir
        self.masks_dir = self.dataset_dir / masks_subdir
        self.output_dir = self.dataset_dir / 'bbox_masks'

        # Verify directories exist
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise ValueError(f"Masks directory not found: {self.masks_dir}")

    def pixelwise_to_bbox(self, mask):
        """
        Convert pixel-wise mask to bounding box coordinates

        Args:
            mask: 2D numpy array with binary mask

        Returns:
            tuple: (x_min, y_min, x_max, y_max) or None if no object found
        """
        # Find all non-zero pixels
        coords = np.argwhere(mask > 0)

        if len(coords) == 0:
            return None  # No object found

        # Get min and max coordinates
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return (x_min, y_min, x_max, y_max)

    def create_box_mask(self, mask, padding=0):
        """
        Create bounding box mask from pixel-wise mask

        Args:
            mask: 2D numpy array with binary mask
            padding: Additional pixels to add around bounding box (default: 0)

        Returns:
            2D numpy array with bounding box mask
        """
        bbox = self.pixelwise_to_bbox(mask)

        if bbox is None:
            return np.zeros_like(mask)

        x_min, y_min, x_max, y_max = bbox

        # Add padding
        if padding > 0:
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(mask.shape[1] - 1, x_max + padding)
            y_max = min(mask.shape[0] - 1, y_max + padding)

        # Create box mask
        box_mask = np.zeros_like(mask)
        box_mask[y_min:y_max+1, x_min:x_max+1] = 255

        return box_mask

    def process_single_mask(self, mask_path, output_path, padding=0, visualize=False):
        """
        Process a single mask file

        Args:
            mask_path: Path to input mask
            output_path: Path to save output bbox mask
            padding: Additional pixels around bbox (default: 0)
            visualize: If True, display comparison (default: False)
        """
        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Warning: Could not read mask: {mask_path}")
            return False

        # Convert to binary (if not already)
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Create bbox mask
        bbox_mask = self.create_box_mask(mask_binary, padding=padding)

        # Save
        cv2.imwrite(str(output_path), bbox_mask)

        # Visualize if requested
        if visualize:
            self.visualize_comparison(mask_binary, bbox_mask, mask_path.name)

        return True

    def process_all(self, padding=0, visualize_samples=0):
        """
        Process all masks in the masks directory

        Args:
            padding: Additional pixels around bbox (default: 0)
            visualize_samples: Number of samples to visualize (default: 0)
        """
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Get all mask files
        mask_files = sorted(list(self.masks_dir.glob('*.png')))

        if len(mask_files) == 0:
            print(f"No PNG masks found in {self.masks_dir}")
            return

        print(f"Found {len(mask_files)} masks to process")
        print(f"Output directory: {self.output_dir}")

        # Process each mask
        successful = 0
        for i, mask_path in enumerate(tqdm(mask_files, desc="Processing masks")):
            output_path = self.output_dir / mask_path.name

            # Visualize only first N samples
            visualize = (i < visualize_samples)

            if self.process_single_mask(mask_path, output_path, padding, visualize):
                successful += 1

        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful}/{len(mask_files)} masks")
        print(f"Bounding box masks saved to: {self.output_dir}")

    def visualize_comparison(self, original_mask, bbox_mask, title=""):
        """Visualize original vs bbox mask"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_mask, cmap='gray')
        axes[0].set_title('Original Pixel-wise Mask')
        axes[0].axis('off')

        axes[1].imshow(bbox_mask, cmap='gray')
        axes[1].set_title('Bounding Box Mask')
        axes[1].axis('off')

        # Overlay
        overlay = cv2.cvtColor(original_mask, cv2.COLOR_GRAY2RGB)
        bbox_colored = np.zeros_like(overlay)
        bbox_colored[bbox_mask > 0] = [255, 0, 0]  # Red for bbox

        alpha = 0.4
        overlay = cv2.addWeighted(overlay, 1-alpha, bbox_colored, alpha, 0)

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Red = BBox)')
        axes[2].axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def get_statistics(self):
        """Get statistics about the bounding boxes"""
        mask_files = list(self.masks_dir.glob('*.png'))

        if len(mask_files) == 0:
            print("No masks found")
            return

        bbox_sizes = []
        aspect_ratios = []
        coverage_ratios = []

        for mask_path in tqdm(mask_files, desc="Analyzing masks"):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            bbox = self.pixelwise_to_bbox(mask_binary)

            if bbox is None:
                continue

            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            bbox_area = width * height
            mask_area = np.sum(mask_binary > 0)

            bbox_sizes.append((width, height))
            aspect_ratios.append(width / height if height > 0 else 0)
            coverage_ratios.append(mask_area / bbox_area if bbox_area > 0 else 0)

        # Print statistics
        print("\n=== Bounding Box Statistics ===")
        print(f"Total masks analyzed: {len(bbox_sizes)}")

        if bbox_sizes:
            widths, heights = zip(*bbox_sizes)
            print(f"\nBBox Width: min={min(widths)}, max={max(widths)}, "
                  f"mean={np.mean(widths):.1f}")
            print(f"BBox Height: min={min(heights)}, max={max(heights)}, "
                  f"mean={np.mean(heights):.1f}")
            print(f"Aspect Ratio (W/H): min={min(aspect_ratios):.2f}, "
                  f"max={max(aspect_ratios):.2f}, mean={np.mean(aspect_ratios):.2f}")
            print(f"Coverage Ratio (mask/bbox): min={min(coverage_ratios):.2%}, "
                  f"max={max(coverage_ratios):.2%}, mean={np.mean(coverage_ratios):.2%}")


# Example usage
if __name__ == "__main__":
    # Configure your paths
    DATASET_DIR = data2path  # Change this to your dataset directory
    IMAGES_SUBDIR = "images"  # Change if different
    MASKS_SUBDIR = "masks"    # Change if different

    # Initialize converter
    converter = PixelwiseToBBoxConverter(
        dataset_dir=DATASET_DIR,
        images_subdir=IMAGES_SUBDIR,
        masks_subdir=MASKS_SUBDIR
    )

    # Option 1: Get statistics about your masks (optional but recommended)
    print("Analyzing masks...")
    converter.get_statistics()

    # Option 2: Process all masks
    # padding=5 adds 5 pixels around the bounding box (set to 0 for tight boxes)
    # visualize_samples=3 shows first 3 comparisons (set to 0 to skip visualization)
    converter.process_all(padding=0, visualize_samples=5)

    # Option 3: Process a single mask for testing
    # mask_path = Path(DATASET_DIR) / MASKS_SUBDIR / "lung_031_slice_267.png"
    # output_path = Path("/content") / "sample_mask.png"
    # converter.process_single_mask(mask_path, output_path, padding=0, visualize=True)