import os
from datasets import load_dataset
from PIL import Image


# Output folder
OUTPUT_DIR="../hagrid_100_data"
os.makedirs(OUTPUT_DIR,exist_ok=True)


# Load Hugging Face HaGRID subset
dataset=load_dataset(
    "GestureDetectionConnoisseurs/hagrid_subsets",
    split="train"
)


# We only want 100 images per class
TARGET_PER_CLASS=100

class_counts={}

total_saved=0


for idx,example in enumerate(dataset):
    image=example["image"]
    label=example["label"]

    # Convert label to string folder name
    label=str(label)

    if label not in class_counts:
        class_counts[label]=0

    if class_counts[label]>=TARGET_PER_CLASS:
        continue

    class_dir=os.path.join(OUTPUT_DIR,label)
    os.makedirs(class_dir,exist_ok=True)

    save_path=os.path.join(
        class_dir,
        f"{label}_{class_counts[label]:04d}.jpg"
    )

    # Make sure image saves as RGB jpg
    if isinstance(image,Image.Image):
        image=image.convert("RGB")
        image.save(save_path)
    else:
        image=Image.fromarray(image).convert("RGB")
        image.save(save_path)

    class_counts[label]+=1
    total_saved+=1

    if total_saved%100==0:
        print(f"Saved {total_saved} images so far...")


print("\nDone creating HaGRID 100-per-class subset.")
print("------------------------------------------")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Total images saved: {total_saved}")

print("\nImages saved per class:")
for label,count in sorted(class_counts.items()):
    print(f"{label}: {count}")
