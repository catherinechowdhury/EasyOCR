import os

# Paths (adjust if needed)
TRAIN_IMG_DIR = "all_data/en/train"
TEST_IMG_DIR  = "all_data/en/test"

TRAIN_LABELS = "all_data/train_labels.txt"
TEST_LABELS  = "all_data/test_labels.txt"

# Load allowed images
train_images = set(os.listdir(TRAIN_IMG_DIR))
test_images = set(os.listdir(TEST_IMG_DIR))


def load_labels(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # MUST be tab-separated
            if "\t" not in line:
                print(f"BAD FORMAT (no TAB): {line}")
                continue

            fname, label = line.split("\t", 1)

            data[fname] = label

    return data


train_labels = load_labels(TRAIN_LABELS)
test_labels  = load_labels(TEST_LABELS)


# -----------------------------
# 1. Check missing images
# -----------------------------
print("\nChecking TRAIN set...")

missing_train = []
for fname in train_labels:
    if fname not in train_images:
        missing_train.append(fname)

print(f"Missing train images: {len(missing_train)}")
if missing_train:
    print(missing_train[:10])


print("\nChecking TEST set...")

missing_test = []
for fname in test_labels:
    if fname not in test_images:
        missing_test.append(fname)

print(f"Missing test images: {len(missing_test)}")
if missing_test:
    print(missing_test[:10])


# -----------------------------
# 2. Check unlabeled images
# -----------------------------
unlabeled_train = train_images - set(train_labels.keys())
unlabeled_test  = test_images - set(test_labels.keys())

print("\n Unlabeled TRAIN images:", len(unlabeled_train))
print(" Unlabeled TEST images:", len(unlabeled_test))


# -----------------------------
# 3. Check empty labels
# -----------------------------
empty_labels = []

for k, v in train_labels.items():
    if v.strip() == "":
        empty_labels.append(k)

for k, v in test_labels.items():
    if v.strip() == "":
        empty_labels.append(k)

print("\n🔍 Empty labels:", len(empty_labels))


# -----------------------------
# 4. Character validation
# -----------------------------
ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -")

invalid_chars = []

for label_dict in [train_labels, test_labels]:
    for fname, label in label_dict.items():
        for ch in label:
            if ch not in ALLOWED:
                invalid_chars.append((fname, ch))

print("\nInvalid characters found:", len(invalid_chars))
if invalid_chars:
    print(invalid_chars[:10])


# -----------------------------
# 5. Summary
# -----------------------------
print("\n====================")
print("DATASET CHECK DONE")
print("====================")

if (missing_train or missing_test or empty_labels or invalid_chars):
    print("Dataset has issues. Fix before training.")
else:
    print(" Dataset looks clean. Ready to train!")