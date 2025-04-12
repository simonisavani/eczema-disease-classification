import os
import shutil
import random
from tqdm import tqdm

original_dir = "datasets"
test_dir = os.path.join(original_dir, "test")
categories = ["Eczema", "Not Eczema"]

# Create test dirs
for cat in categories:
    os.makedirs(os.path.join(test_dir, cat), exist_ok=True)

for category in categories:
    source_dir = os.path.join(original_dir, category)
    dest_dir = os.path.join(test_dir, category)

    all_images = os.listdir(source_dir)
    test_sample = random.sample(all_images, int(len(all_images) * 0.15))  # 15% for testing

    print(f"Copying {len(test_sample)} images to test/{category}...")

    for img in tqdm(test_sample):
        shutil.copy2(os.path.join(source_dir, img), os.path.join(dest_dir, img))
