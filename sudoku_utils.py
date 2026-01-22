"""
Sudoku Utilities Module
-----------------------
Contains Sudoku generation, image fetching, and preprocessing utilities
for the Perception Sudoku game.
"""

import numpy as np
import random
import h5py
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import cv2


class SudokuGenerator:
    """Generates valid Sudoku puzzles using backtracking algorithm."""

    def __init__(self):
        self.grid = [[0] * 9 for _ in range(9)]

    def _is_valid(self, grid: List[List[int]], row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid."""
        # Check row
        if num in grid[row]:
            return False

        # Check column - optimized to avoid repeated checks
        for i in range(9):
            if grid[i][col] == num:
                return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if grid[i][j] == num:
                    return False

        return True

    def _solve(self, grid: List[List[int]]) -> bool:
        """Solve the Sudoku using backtracking."""
        for row in range(9):
            for col in range(9):
                if grid[row][col] == 0:
                    nums = list(range(1, 10))
                    random.shuffle(nums)
                    for num in nums:
                        if self._is_valid(grid, row, col, num):
                            grid[row][col] = num
                            if self._solve(grid):
                                return True
                            grid[row][col] = 0
                    return False
        return True

    def _generate_solved_grid(self) -> List[List[int]]:
        """Generate a complete valid Sudoku grid."""
        grid = [[0] * 9 for _ in range(9)]
        self._solve(grid)
        return grid

    def generate_puzzle(self, num_clues: int = 40) -> Tuple[List[List[int]], List[List[int]], List[Tuple[int, int]]]:
        """
        Generate a Sudoku puzzle.

        Args:
            num_clues: Number of clues to keep (default ~40, removing ~41)

        Returns:
            puzzle: The puzzle with empty cells (0)
            solution: The complete solution
            clue_positions: List of (row, col) tuples for clue cells
        """
        solution = self._generate_solved_grid()
        puzzle = [row[:] for row in solution]  # Deep copy

        # Get all positions and shuffle
        all_positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(all_positions)

        # Remove cells to create puzzle
        cells_to_remove = 81 - num_clues
        for i in range(cells_to_remove):
            row, col = all_positions[i]
            puzzle[row][col] = 0

        # Get clue positions (remaining positions after removal)
        # Since positions are shuffled, clue positions are simply the ones not removed
        clue_positions = all_positions[cells_to_remove:]

        return puzzle, solution, clue_positions


class ImageFetcher:
    """Handles loading images from SVHN dataset or generating synthetic fallbacks."""

    def __init__(self, data_path: str = "svhn_data.h5"):
        self.data_path = data_path
        self.images_by_digit: Dict[int, np.ndarray] = {}
        self.using_synthetic = False
        self._load_data()

    def _load_data(self):
        """Load SVHN data or generate synthetic fallback."""
        if os.path.exists(self.data_path):
            try:
                with h5py.File(self.data_path, 'r') as f:
                    # Try common SVHN h5 structures
                    if 'X_train' in f and 'y_train' in f:
                        images = f['X_train'][:]
                        labels = f['y_train'][:]
                    elif 'images' in f and 'labels' in f:
                        images = f['images'][:]
                        labels = f['labels'][:]
                    elif 'X' in f and 'y' in f:
                        images = f['X'][:]
                        labels = f['y'][:]
                    else:
                        # Try to get first two datasets
                        keys = list(f.keys())
                        if len(keys) >= 2:
                            images = f[keys[0]][:]
                            labels = f[keys[1]][:]
                        else:
                            raise KeyError("Could not find image/label datasets")

                    # Flatten labels if needed
                    labels = labels.flatten()

                    # Group images by digit (1-9, ignoring 0 for Sudoku)
                    for digit in range(1, 10):
                        digit_indices = np.where(labels == digit)[0]
                        if len(digit_indices) > 0:
                            self.images_by_digit[digit] = images[digit_indices]

                    if self.images_by_digit:
                        print(f"Loaded SVHN data: {sum(len(v) for v in self.images_by_digit.values())} images")
                        return

            except Exception as e:
                print(f"Error loading SVHN data: {e}")

        # Fallback to synthetic noise images
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate synthetic noise images with digit-like patterns."""
        print("SVHN data not found. Generating synthetic noise images...")
        self.using_synthetic = True

        for digit in range(1, 10):
            # Generate 100 synthetic images per digit
            images = []
            for _ in range(100):
                img = self._create_noisy_digit_image(digit)
                images.append(img)
            self.images_by_digit[digit] = np.array(images)

    def _create_noisy_digit_image(self, digit: int) -> np.ndarray:
        """Create a synthetic 32x32 noisy image with a digit-like pattern."""
        # Start with noise
        img = np.random.randint(0, 80, (32, 32, 3), dtype=np.uint8)

        # Add a brighter region in the center resembling a digit
        center_y, center_x = 16, 16

        # Draw a simple representation of the digit
        try:
            cv2.putText(
                img,
                str(digit),
                (8, 26),  # Position
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,  # Font scale
                (200 + random.randint(-30, 30),
                 200 + random.randint(-30, 30),
                 200 + random.randint(-30, 30)),  # Color with variation
                2,  # Thickness
                cv2.LINE_AA
            )
        except:
            # If cv2.putText fails, just add noise patterns
            pass

        # Add more noise
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Random rotation and slight distortion
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((16, 16), angle, 1.0)
        img = cv2.warpAffine(img, M, (32, 32))

        return img

    def get_image_for_digit(self, digit: int) -> np.ndarray:
        """
        Get a random image for the given digit.

        Args:
            digit: The digit (1-9)

        Returns:
            A 32x32 image (may be RGB or grayscale depending on source)
        """
        if digit not in self.images_by_digit or len(self.images_by_digit[digit]) == 0:
            # Return pure noise if digit not available
            return np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        idx = np.random.randint(0, len(self.images_by_digit[digit]))
        return self.images_by_digit[digit][idx].copy()


class ImagePreprocessor:
    """Preprocesses images for the CNN model."""

    @staticmethod
    def preprocess_for_model(image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for the CNN model.

        Args:
            image: Input image (any shape)

        Returns:
            Preprocessed image with shape (1, 32, 32, 1)
        """
        # Ensure we're working with a numpy array
        img = np.array(image)

        # Resize to 32x32 if needed
        if img.shape[:2] != (32, 32):
            img = cv2.resize(img, (32, 32))

        # Convert to grayscale - optimized shape checking
        if len(img.shape) == 3:
            channels = img.shape[2]
            if channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif channels == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif channels == 1:
                img = img.squeeze()

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Expand dimensions for model: (1, 32, 32, 1)
        img = np.expand_dims(img, axis=-1)  # (32, 32, 1)
        img = np.expand_dims(img, axis=0)   # (1, 32, 32, 1)

        return img

    @staticmethod
    def prepare_for_display(image: np.ndarray) -> np.ndarray:
        """
        Prepare an image for display in Streamlit.

        Args:
            image: Input image

        Returns:
            Image suitable for st.image() (values in 0-255 range, uint8)
        """
        img = np.array(image)

        # Handle different input shapes
        if len(img.shape) == 4:
            img = img.squeeze()

        # Ensure 0-255 range
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Convert grayscale to RGB for better display - consolidated logic
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            # Squeeze if needed before conversion
            img_2d = img.squeeze() if len(img.shape) == 3 else img
            img = cv2.cvtColor(img_2d, cv2.COLOR_GRAY2RGB)

        return img


def validate_sudoku_move(grid: List[List[int]], row: int, col: int, num: int) -> bool:
    """
    Check if a move is valid in Sudoku.

    Args:
        grid: Current grid state
        row: Row index
        col: Column index
        num: Number to place (1-9)

    Returns:
        True if the move is valid
    """
    if num < 1 or num > 9:
        return False

    # Temporarily remove the current value
    original = grid[row][col]
    grid[row][col] = 0

    # Check row
    if num in grid[row]:
        grid[row][col] = original
        return False

    # Check column - optimized to avoid list comprehension
    for i in range(9):
        if grid[i][col] == num:
            grid[row][col] = original
            return False

    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            if grid[i][j] == num:
                grid[row][col] = original
                return False

    grid[row][col] = original
    return True


def check_solution(user_grid: List[List[int]], solution: List[List[int]],
                   predictions: Dict[Tuple[int, int], int]) -> Tuple[bool, List[Dict]]:
    """
    Check the user's solution against the ground truth.

    Args:
        user_grid: The user's completed grid
        solution: The correct solution
        predictions: Dictionary of {(row, col): predicted_digit} for clue cells

    Returns:
        (is_correct, errors): Boolean and list of error dictionaries
    """
    errors = []

    for row in range(9):
        for col in range(9):
            user_val = user_grid[row][col]
            correct_val = solution[row][col]

            if user_val != correct_val:
                error_info = {
                    'row': row,
                    'col': col,
                    'user_value': user_val,
                    'correct_value': correct_val,
                    'is_hallucination': False
                }

                # Check if this was an AI hallucination
                if (row, col) in predictions:
                    predicted = predictions[(row, col)]
                    if predicted != correct_val and user_val == predicted:
                        error_info['is_hallucination'] = True
                        error_info['ai_predicted'] = predicted

                errors.append(error_info)

    return len(errors) == 0, errors
