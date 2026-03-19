# ============================================================
# GLOBALMEDNET — PHASE 1: DICOM INGESTION PIPELINE
# ============================================================
# Paste each CELL block into a separate Kaggle notebook cell.
# Run them top to bottom, one at a time.
# ============================================================


# ── CELL 1 ── Install libraries
# Kaggle has most things pre-installed, but we need a couple extras.
# This takes about 60 seconds. You'll see lots of text scroll by — that's normal.

!pip install pydicom monai opencv-python-headless --quiet


# ── CELL 2 ── Import everything we need
# Think of this as opening all the toolboxes before starting work.

import os
import glob
import random
import numpy as np
import pandas as pd
import pydicom                          # reads DICOM medical image files
import cv2                              # image processing
import matplotlib.pyplot as plt         # for displaying images
import torch                            # the AI framework we'll use later
from pathlib import Path
from monai.transforms import (          # MONAI = medical AI toolkit by NVIDIA
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    Resize,
    RandFlip,
    RandRotate,
    RandZoom,
    ToTensor,
)
from torch.utils.data import Dataset, DataLoader

print("✅ All libraries imported successfully")
print(f"   PyTorch version: {torch.__version__}")
print(f"   GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")


# ── CELL 3 ── Find our data
# Let's see what the RSNA dataset looks like and where the files live.

RSNA_DIR = "/kaggle/input/rsna-pneumonia-detection-challenge"

# List everything in the dataset folder
print("📁 Dataset contents:")
for item in sorted(os.listdir(RSNA_DIR)):
    full_path = os.path.join(RSNA_DIR, item)
    if os.path.isdir(full_path):
        count = len(os.listdir(full_path))
        print(f"   📂 {item}/  ({count} files)")
    else:
        size_mb = os.path.getsize(full_path) / 1e6
        print(f"   📄 {item}  ({size_mb:.1f} MB)")


# ── CELL 4 ── Load the labels CSV
# The dataset comes with a CSV file telling us which patients have pneumonia.
# 1 = pneumonia detected, 0 = normal

labels_df = pd.read_csv(f"{RSNA_DIR}/stage_2_train_labels.csv")

print("📊 Label file loaded:")
print(f"   Total rows: {len(labels_df):,}")
print(f"   Columns: {list(labels_df.columns)}")
print()
print(labels_df.head(10))
print()

# Count how many normal vs pneumonia cases
value_counts = labels_df.drop_duplicates('patientId')['Target'].value_counts()
print("🏥 Class distribution:")
print(f"   Normal (0):    {value_counts.get(0, 0):,} patients")
print(f"   Pneumonia (1): {value_counts.get(1, 0):,} patients")


# ── CELL 5 ── Open and inspect one DICOM file
# A DICOM file is the standard medical image format.
# It contains the image PLUS metadata (patient age, scanner type, etc.)
# We never use real patient data — but we can see what fields exist.

train_dir = f"{RSNA_DIR}/stage_2_train_images"
sample_files = glob.glob(f"{train_dir}/*.dcm")

# Open one file
sample_dcm = pydicom.dcmread(sample_files[0])

print("🔬 Sample DICOM metadata (anonymised fields only):")
print(f"   Image size:      {sample_dcm.Rows} x {sample_dcm.Columns} pixels")
print(f"   Pixel type:      {sample_dcm.pixel_array.dtype}")
print(f"   Pixel range:     {sample_dcm.pixel_array.min()} – {sample_dcm.pixel_array.max()}")
print(f"   Modality:        {getattr(sample_dcm, 'Modality', 'N/A')}")
print(f"   Body part:       {getattr(sample_dcm, 'BodyPartExamined', 'N/A')}")
print()
print(f"   Raw pixel array shape: {sample_dcm.pixel_array.shape}")


# ── CELL 6 ── Visualise a few raw X-rays
# Let's actually look at some of the images before we process them.
# This is what the AI will be learning from.

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Raw DICOM X-rays — before processing", fontsize=14, y=1.02)

# Get 4 normal and 4 pneumonia cases
patient_labels = labels_df.drop_duplicates('patientId')
normal_ids    = patient_labels[patient_labels['Target'] == 0]['patientId'].tolist()
pneumonia_ids = patient_labels[patient_labels['Target'] == 1]['patientId'].tolist()

samples = [('Normal', normal_ids[:4]), ('Pneumonia', pneumonia_ids[:4])]

for row_idx, (label, ids) in enumerate(samples):
    for col_idx, pid in enumerate(ids):
        dcm_path = f"{train_dir}/{pid}.dcm"
        if not os.path.exists(dcm_path):
            continue
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(float)
        # Normalise brightness for display
        img = (img - img.min()) / (img.max() - img.min())
        axes[row_idx][col_idx].imshow(img, cmap='gray')
        axes[row_idx][col_idx].set_title(f"{label}", fontsize=10)
        axes[row_idx][col_idx].axis('off')

plt.tight_layout()
plt.savefig('raw_xrays.png', dpi=100, bbox_inches='tight')
plt.show()
print("✅ Raw X-rays displayed. Top row = Normal. Bottom row = Pneumonia.")


# ── CELL 7 ── The preprocessing function
# Raw DICOM images can't go straight into an AI model. We need to:
#   1. Read the DICOM pixel data
#   2. Resize every image to the same size (224x224)
#   3. Normalise pixel values to a 0–1 range
#   4. Convert to a format PyTorch understands (a "tensor")
# This function does all of that for a single image.

TARGET_SIZE = 224  # standard size for most medical AI models

def preprocess_dicom(dcm_path: str) -> np.ndarray:
    """
    Reads one DICOM file and returns a clean numpy array.
    Shape will be (224, 224) — ready for model input.
    """
    # Step 1: Read the DICOM file
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)

    # Step 2: Handle edge case where image has 3 channels (very rare in RSNA)
    if len(img.shape) == 3:
        img = img[:, :, 0]

    # Step 3: Resize to TARGET_SIZE x TARGET_SIZE
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    # Step 4: Normalise to [0, 1] range
    # Some scanners use different brightness scales — this standardises them
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)  # blank image fallback

    return img


def apply_augmentation(img: np.ndarray, is_training: bool = True) -> np.ndarray:
    """
    Data augmentation: randomly modify training images so the model
    doesn't memorise exact pixel patterns. Only applied during training.
    Techniques: horizontal flip, small rotation, slight zoom.
    """
    if not is_training:
        return img

    # Random horizontal flip (chest X-rays are symmetric)
    if random.random() > 0.5:
        img = np.fliplr(img)

    # Random rotation ±7 degrees (small, realistic for patient positioning)
    if random.random() > 0.5:
        angle = random.uniform(-7, 7)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)

    # Random zoom ±10%
    if random.random() > 0.5:
        scale = random.uniform(0.9, 1.1)
        h, w = img.shape
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        # Crop or pad back to original size
        if scale > 1.0:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            img = img_resized[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img = np.pad(img_resized, ((pad_h, h-new_h-pad_h),
                                        (pad_w, w-new_w-pad_w)), mode='reflect')

    return img.astype(np.float32)


print("✅ Preprocessing functions defined")
print(f"   Target image size: {TARGET_SIZE}x{TARGET_SIZE} pixels")
print("   Augmentations: horizontal flip, ±7° rotation, ±10% zoom")


# ── CELL 8 ── Test the preprocessor on real images
# Let's run a few images through the pipeline and see what they look like after.

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f"After preprocessing — {TARGET_SIZE}x{TARGET_SIZE}, normalised to [0,1]",
             fontsize=13, y=1.02)

for row_idx, (label, ids) in enumerate(samples):
    for col_idx, pid in enumerate(ids[:4]):
        dcm_path = f"{train_dir}/{pid}.dcm"
        if not os.path.exists(dcm_path):
            continue
        processed = preprocess_dicom(dcm_path)
        axes[row_idx][col_idx].imshow(processed, cmap='gray')
        axes[row_idx][col_idx].set_title(
            f"{label}\nmin={processed.min():.2f} max={processed.max():.2f}",
            fontsize=9
        )
        axes[row_idx][col_idx].axis('off')

plt.tight_layout()
plt.savefig('processed_xrays.png', dpi=100, bbox_inches='tight')
plt.show()
print("✅ All images are now the same size and normalised.")


# ── CELL 9 ── Visualise augmentation
# Let's show the same X-ray 8 times with different random augmentations.
# This is what the model sees during training — slightly different each time.

sample_pid = pneumonia_ids[0]
sample_path = f"{train_dir}/{sample_pid}.dcm"
base_img = preprocess_dicom(sample_path)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Same X-ray, 8 different augmentations (what the AI sees during training)",
             fontsize=12, y=1.02)

for i, ax in enumerate(axes.flat):
    augmented = apply_augmentation(base_img.copy(), is_training=True)
    ax.imshow(augmented, cmap='gray')
    ax.set_title(f"Version {i+1}", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('augmented_xrays.png', dpi=100, bbox_inches='tight')
plt.show()
print("✅ Augmentation working. Each training epoch the model sees fresh variants.")


# ── CELL 10 ── Build the Dataset class
# PyTorch needs data wrapped in a "Dataset" object.
# Think of this as a smart list that loads and processes images on demand
# instead of loading everything into RAM at once.

class RSNAChestXrayDataset(Dataset):
    """
    PyTorch Dataset for RSNA Pneumonia Detection.
    Loads DICOM files on-the-fly, preprocesses them, and returns
    (image_tensor, label) pairs ready for model training.
    """

    def __init__(self, patient_ids: list, labels_dict: dict,
                 dicom_dir: str, is_training: bool = True):
        """
        patient_ids  : list of patient ID strings
        labels_dict  : dict mapping patient_id -> 0 or 1
        dicom_dir    : path to folder containing .dcm files
        is_training  : if True, applies augmentation
        """
        # Only keep patients whose DICOM file actually exists
        self.samples = []
        for pid in patient_ids:
            dcm_path = os.path.join(dicom_dir, f"{pid}.dcm")
            if os.path.exists(dcm_path):
                label = labels_dict.get(pid, 0)
                self.samples.append((dcm_path, label))

        self.is_training = is_training
        print(f"   Dataset created: {len(self.samples):,} images, "
              f"training={is_training}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dcm_path, label = self.samples[idx]

        # Load and preprocess
        img = preprocess_dicom(dcm_path)

        # Apply augmentation if training
        img = apply_augmentation(img, self.is_training)

        # Add channel dimension: (224, 224) → (1, 224, 224)
        # Most AI models expect (channels, height, width)
        # Chest X-rays are grayscale so channels = 1
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor


print("✅ Dataset class defined")


# ── CELL 11 ── Split data into train / validation / test sets
# We split our data three ways:
#   Train (70%)      — the model learns from this
#   Validation (15%) — we check accuracy during training (model never trains on this)
#   Test (15%)       — final evaluation, used for the demo video
#
# We NEVER let the model train on validation or test data.
# That would be cheating and give false accuracy numbers.

# Build a dictionary: patient_id -> label
patient_labels_df = labels_df.drop_duplicates('patientId')
labels_dict = dict(zip(patient_labels_df['patientId'],
                       patient_labels_df['Target']))

all_patient_ids = list(labels_dict.keys())
random.seed(42)  # fixed seed = reproducible splits every time
random.shuffle(all_patient_ids)

n_total = len(all_patient_ids)
n_train = int(0.70 * n_total)
n_val   = int(0.15 * n_total)

train_ids = all_patient_ids[:n_train]
val_ids   = all_patient_ids[n_train:n_train + n_val]
test_ids  = all_patient_ids[n_train + n_val:]

print("📊 Data split:")
print(f"   Training:   {len(train_ids):,} patients  (70%)")
print(f"   Validation: {len(val_ids):,}  patients  (15%)")
print(f"   Test:       {len(test_ids):,}  patients  (15%) ← used for demo video")
print()

# Count class balance in each split
for split_name, split_ids in [("Train", train_ids), ("Val", val_ids), ("Test", test_ids)]:
    n_pos = sum(1 for pid in split_ids if labels_dict.get(pid, 0) == 1)
    n_neg = len(split_ids) - n_pos
    print(f"   {split_name}: {n_neg} normal, {n_pos} pneumonia")


# ── CELL 12 ── Create DataLoaders
# A DataLoader wraps the Dataset and handles:
#   - Loading images in parallel (num_workers)
#   - Batching (sending 32 images at a time to the model)
#   - Shuffling (so the model doesn't see data in the same order twice)

BATCH_SIZE = 32   # 32 images processed at once — good for T4 GPU

print("🔧 Creating datasets...")
train_dataset = RSNAChestXrayDataset(train_ids, labels_dict, train_dir, is_training=True)
val_dataset   = RSNAChestXrayDataset(val_ids,   labels_dict, train_dir, is_training=False)
test_dataset  = RSNAChestXrayDataset(test_ids,  labels_dict, train_dir, is_training=False)

print("\n🔧 Creating DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"\n✅ DataLoaders ready:")
print(f"   Train batches:      {len(train_loader):,}")
print(f"   Validation batches: {len(val_loader):,}")
print(f"   Test batches:       {len(test_loader):,}")
print(f"   Batch size:         {BATCH_SIZE} images")


# ── CELL 13 ── Pipeline smoke test
# Let's pull one batch from the DataLoader and check everything looks right.
# If this cell runs without errors, the pipeline is working end-to-end.

print("🔍 Running pipeline smoke test...")
batch_images, batch_labels = next(iter(train_loader))

print(f"\n   Batch image tensor shape: {batch_images.shape}")
print(f"   Expected:                 torch.Size([{BATCH_SIZE}, 1, 224, 224])")
print(f"   ✅ Match: {batch_images.shape == torch.Size([BATCH_SIZE, 1, 224, 224])}")
print()
print(f"   Pixel value range: {batch_images.min():.4f} – {batch_images.max():.4f}")
print(f"   Expected range:    0.0 – 1.0")
print()
print(f"   Labels in batch: {batch_labels.tolist()[:16]} ...")
print(f"   Label dtype: {batch_labels.dtype}")
print()
print(f"   Memory used by one batch: "
      f"{batch_images.element_size() * batch_images.nelement() / 1e6:.1f} MB")

if torch.cuda.is_available():
    gpu_mem = torch.cuda.memory_allocated() / 1e6
    print(f"   GPU memory allocated: {gpu_mem:.1f} MB")


# ── CELL 14 ── Visualise a processed batch
# Final visual check — show a grid of what the model will actually receive.

fig, axes = plt.subplots(4, 8, figsize=(20, 10))
fig.suptitle(
    f"Processed batch ready for model input — shape {tuple(batch_images.shape)}\n"
    "Green border = Pneumonia | No border = Normal",
    fontsize=12
)

for i, ax in enumerate(axes.flat):
    if i >= len(batch_images):
        ax.axis('off')
        continue
    img = batch_images[i, 0].numpy()   # remove channel dim for display
    label = batch_labels[i].item()
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    if label == 1:
        for spine in ax.spines.values():
            spine.set_edgecolor('#1D9E75')
            spine.set_linewidth(3)
            spine.set_visible(True)

plt.tight_layout()
plt.savefig('model_ready_batch.png', dpi=80, bbox_inches='tight')
plt.show()
print("✅ Batch visualised. Green border = Pneumonia.")


# ── CELL 15 ── Save pipeline outputs for Phase 2
# Save the test patient IDs and labels so Phase 2 can use the same split.
# Also save a small sample of processed images as numpy arrays.

import json

# Save the split IDs
splits = {
    'train_ids': train_ids,
    'val_ids':   val_ids,
    'test_ids':  test_ids,
}
with open('data_splits.json', 'w') as f:
    json.dump(splits, f)

# Save labels dict
with open('labels_dict.json', 'w') as f:
    json.dump(labels_dict, f)

# Save a small sample of processed test images for demo use
print("💾 Saving demo samples...")
demo_samples = []
demo_labels  = []
for pid in test_ids[:50]:  # save first 50 test images
    dcm_path = f"{train_dir}/{pid}.dcm"
    if os.path.exists(dcm_path):
        img = preprocess_dicom(dcm_path)
        demo_samples.append(img)
        demo_labels.append(labels_dict.get(pid, 0))

demo_array = np.stack(demo_samples)
np.save('demo_images.npy',  demo_array)
np.save('demo_labels.npy',  np.array(demo_labels))

print(f"\n✅ Phase 1 complete! Files saved:")
print(f"   data_splits.json      — train/val/test patient ID lists")
print(f"   labels_dict.json      — patient ID → label mapping")
print(f"   demo_images.npy       — {len(demo_samples)} preprocessed test images")
print(f"   demo_labels.npy       — corresponding labels")
print(f"   raw_xrays.png         — visualisation of raw DICOMs")
print(f"   processed_xrays.png   — visualisation after preprocessing")
print(f"   augmented_xrays.png   — augmentation examples")
print(f"   model_ready_batch.png — final batch ready for model input")
print()
print("🚀 Ready for Phase 2: Model Training")


# ── CELL 16 ── Phase 1 summary
print("=" * 60)
print("  GLOBALMEDNET — PHASE 1 COMPLETE")
print("=" * 60)
print()
print("  What we built:")
print("  ✅ DICOM file reader (pydicom)")
print("  ✅ Image preprocessor (resize → normalise → tensor)")
print("  ✅ Data augmentation (flip, rotate, zoom)")
print("  ✅ PyTorch Dataset class")
print("  ✅ Train / Validation / Test split")
print("  ✅ DataLoaders with batching and parallel loading")
print("  ✅ End-to-end pipeline smoke test")
print("  ✅ Demo images saved for the LinkedIn video")
print()
print("  Data processed:")
print(f"  📦 {len(train_dataset):,} training images")
print(f"  📦 {len(val_dataset):,}  validation images")
print(f"  📦 {len(test_dataset):,}  test images (held out for demo)")
print()
print("  Next: Phase 2 — Train the AI model on these images")
print("=" * 60)
