import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

DATASET_DIR = "datasets"
classes = os.listdir(DATASET_DIR)
image_shapes = []

print(f"\n🔍 Checking dataset at: {DATASET_DIR}\n")

for label in classes:
    class_path = os.path.join(DATASET_DIR, label)
    files = os.listdir(class_path)
    print(f"📁 Class '{label}': {len(files)} images")

    for file in files:
        try:
            img_path = os.path.join(class_path, file)
            with Image.open(img_path) as img:
                img.verify()  # check if corrupted
                img = Image.open(img_path)  # reopen after verify
                image_shapes.append(img.size)
        except Exception as e:
            print(f"❌ Corrupted file: {img_path} — {e}")

# Optional: visualize distribution of image shapes
shape_counts = Counter(image_shapes)
top_shapes = shape_counts.most_common(5)
print("\n📏 Top image shapes:")
for shape, count in top_shapes:
    print(f"{shape}: {count} images")

# Plot shape distribution
if top_shapes:
    labels, counts = zip(*top_shapes)
    labels = [f"{w}x{h}" for (w, h) in labels]
    plt.bar(labels, counts)
    plt.title("Top Image Shapes")
    plt.ylabel("Image Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# import os
# from PIL import Image

# DATASET_DIR = "datasets"
# TARGET_SIZE = (224, 224)

# def remove_and_resize_images():
#     for label in os.listdir(DATASET_DIR):
#         class_dir = os.path.join(DATASET_DIR, label)
#         for img_name in os.listdir(class_dir):
#             img_path = os.path.join(class_dir, img_name)
#             try:
#                 with Image.open(img_path) as img:
#                     # Convert to RGB if the image is in RGBA mode
#                     if img.mode == 'RGBA':
#                         img = img.convert('RGB')
#                     # Resize image to the target size
#                     img = img.resize(TARGET_SIZE)
#                     img.save(img_path)  # Save resized image
#             except Exception as e:
#                 print(f"❌ Error processing {img_path}: {e}")
#                 # Remove the problematic image
#                 os.remove(img_path)
#                 print(f"✅ Removed problematic image: {img_path}")

# remove_and_resize_images()
# print(f"✅ All images resized to {TARGET_SIZE}.")

# import os
# from PIL import Image
# import numpy as np
# from torchvision import transforms
# from torchvision.transforms import functional as F

# DATASET_DIR = "datasets"
# TARGET_SIZE = (224, 224)

# # Define augmentation transformations
# augmentation = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(20),
#     transforms.RandomResizedCrop(224),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     transforms.RandomVerticalFlip(),
# ])

# def augment_images():
#     for label in os.listdir(DATASET_DIR):
#         class_dir = os.path.join(DATASET_DIR, label)
#         for img_name in os.listdir(class_dir):
#             img_path = os.path.join(class_dir, img_name)
#             try:
#                 with Image.open(img_path) as img:
#                     # Convert to RGB if the image is in RGBA mode
#                     if img.mode == 'RGBA':
#                         img = img.convert('RGB')
#                     # Apply augmentation
#                     img = augmentation(img)
#                     # Save the augmented image (you may save it to a new folder)
#                     img.save(f"augmented_{img_name}")  # For example, save to a new file name
#             except Exception as e:
#                 print(f"❌ Error augmenting {img_path}: {e}")

# augment_images()
# print(f"✅ Augmentation completed for 'Not Eczema'.")
