import os
import shutil
from pathlib import Path

def create_yolo_structure(base_path, dataset_name):
    # Define paths
    base_path = Path(base_path)
    images_path = base_path / 'images'
    labels_path = base_path / 'labels'
    yolo_path = base_path / dataset_name

    # Create YOLO dataset structure
    yolo_images_train = yolo_path / 'images' / 'train'
    yolo_images_val = yolo_path / 'images' / 'val'
    yolo_labels_train = yolo_path / 'labels' / 'train'
    yolo_labels_val = yolo_path / 'labels' / 'val'

    # Create directories
    for path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
        path.mkdir(parents=True, exist_ok=True)

    # Split dataset (e.g., 80% train, 20% val)
    image_files = list(images_path.rglob('*.jpg')) + list(images_path.rglob('*.png'))  # Support .jpg and .png
    label_files = list(labels_path.rglob('*.txt'))

    # Ensure matching images and labels
    images = {img.stem: img for img in image_files}
    labels = {lbl.stem: lbl for lbl in label_files}
    common_files = set(images.keys()).intersection(labels.keys())

    if not common_files:
        print("No matching images and labels found. Please check your dataset.")
        print(f"Images found: {len(image_files)}, Labels found: {len(label_files)}")
        return

    images = [images[stem] for stem in common_files]
    labels = [labels[stem] for stem in common_files]

    split_idx = int(0.8 * len(images))
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    # Copy files
    for img, lbl in zip(train_images, train_labels):
        shutil.copy(img, yolo_images_train / img.name)
        shutil.copy(lbl, yolo_labels_train / lbl.name)

    for img, lbl in zip(val_images, val_labels):
        shutil.copy(img, yolo_images_val / img.name)
        shutil.copy(lbl, yolo_labels_val / lbl.name)

    # Debugging output
    print(f"Train images: {len(train_images)}, Validation images: {len(val_images)}")
    print(f"Train labels: {len(train_labels)}, Validation labels: {len(val_labels)}")

    # Create data.yaml
    yaml_content = f"""
        train: {yolo_images_train.as_posix()}
        val: {yolo_images_val.as_posix()}

        nc: 1
        names: ['Wooden Box']
        """

    with open(yolo_path / 'data.yaml', 'w') as yaml_file:
        yaml_file.write(yaml_content)

    print(f"YOLO dataset structure created at {yolo_path}")

# Example usage
create_yolo_structure(base_path='datasets_new', dataset_name='yolo_dataset_new')
