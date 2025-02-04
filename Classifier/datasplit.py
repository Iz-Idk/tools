import os
import shutil
import random

import os
import shutil
import random

def split_dataset(source_dir, dest_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train, validation, and test sets.

    Args:
        source_dir (str): Path to the source directory containing the images.
        dest_dir (str): Path to the destination directory where the splits will be saved.
        train_ratio (float): Fraction of data to use for training.
        val_ratio (float): Fraction of data to use for validation.
        test_ratio (float): Fraction of data to use for testing.
    """
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Directories for train, validation, and test
    train_dir = os.path.join(dest_dir, 'train/Smokes')
    val_dir = os.path.join(dest_dir, 'val/Smokes')
    test_dir = os.path.join(dest_dir, 'test/Smokes')

    for dir in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Get all image files from the source directory
    images = [f for f in os.listdir(source_dir) if (os.path.isfile(os.path.join(source_dir, f)) )] # and "NoSmokes_camera" in f

    # Shuffle the images randomly
    random.shuffle(images)

    # Calculate the number of images for each set
    total_images = len(images)
    train_size = int(train_ratio * total_images)
    val_size = int(val_ratio * total_images)
    test_size = total_images - train_size - val_size

    # Split the images into the respective sets
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    # Move images into the corresponding directories
    for img in train_images:
        shutil.move(os.path.join(source_dir, img), os.path.join(train_dir, img))

    for img in val_images:
        shutil.move(os.path.join(source_dir, img), os.path.join(val_dir, img))

    for img in test_images:
        shutil.move(os.path.join(source_dir, img), os.path.join(test_dir, img))

    print(f"Dataset split completed: {train_size} for training, {val_size} for validation, {test_size} for testing.")

# Example usage
source_directory = "C:/Users/Spacelab3/Desktop/envs/Classifier/Smokes"
destination_directory = "C:/Users/Spacelab3/Desktop/envs/Classifier/BalDataset"

split_dataset(source_directory, destination_directory)
