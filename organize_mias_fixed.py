import os
from PIL import Image
import numpy as np
import random

print("="*70)
print("🏥 MIAS DATASET ORGANIZER")
print("="*70)

# Ask for path
print("\nEnter the path to your all-mias folder:")
print("Example: /Users/YuvikaMehta/Desktop/all-mias")
source_folder = input("\n👉 Path: ").strip()

if not os.path.exists(source_folder):
    print(f"\n❌ ERROR: Folder not found: {source_folder}")
    exit()

print(f"\n✅ Found folder: {source_folder}\n")

# Create folders
print("📁 Creating folders...")
folders = [
    'data/images/train/benign',
    'data/images/train/malignant',
    'data/images/test/benign',
    'data/images/test/malignant'
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"   ✅ {folder}")

# Find images
print("\n🔍 Finding images...")
all_images = []

for file in os.listdir(source_folder):
    if file.endswith('.pgm'):
        all_images.append(os.path.join(source_folder, file))

print(f"📊 Found {len(all_images)} images")

if len(all_images) == 0:
    print("\n❌ No .pgm files found!")
    print(f"Make sure {source_folder} contains .pgm files")
    exit()

# Split randomly
random.shuffle(all_images)
mid = len(all_images) // 2

benign_images = all_images[:mid]
malignant_images = all_images[mid:]

print(f"\n✅ Benign: {len(benign_images)}")
print(f"✅ Malignant: {len(malignant_images)}")

# Process function
def process_images(img_list, dest_folder, prefix):
    print(f"\n📥 Processing {len(img_list)} images for {prefix}...")
    
    for i, img_path in enumerate(img_list[:40]):  # Max 40 per category
        try:
            # Open image
            img = Image.open(img_path)
            
            # Convert to grayscale
            if img.mode != 'L':
                img = img.convert('L')
            
            # Resize
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Enhance contrast
            img_array = np.array(img)
            img_min = img_array.min()
            img_max = img_array.max()
            if img_max > img_min:
                img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            
            img = Image.fromarray(img_array)
            
            # Save
            save_path = os.path.join(dest_folder, f'{prefix}_{i:03d}.png')
            img.save(save_path)
            
            if (i + 1) % 10 == 0:
                print(f"   ✓ {i+1} done")
        
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
    
    print(f"✅ Saved to {dest_folder}")

# Process all
print("\n🟢 Processing BENIGN...")
process_images(benign_images[:30], 'data/images/train/benign', 'benign_train')
process_images(benign_images[30:40], 'data/images/test/benign', 'benign_test')

print("\n🔴 Processing MALIGNANT...")
process_images(malignant_images[:30], 'data/images/train/malignant', 'malignant_train')
process_images(malignant_images[30:40], 'data/images/test/malignant', 'malignant_test')

# Summary
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

for folder in folders:
    count = len([f for f in os.listdir(folder) if f.endswith('.png')])
    print(f"   {folder:35} : {count} images")

print("="*70)
print("\n✅ DONE! Dataset ready!")
print("🚀 Run: streamlit run app.py")