import torch
import matplotlib.pyplot as plt
import random

from constants import MODEL_WEIGHTS_PATH, CIFAR_MEAN, CIFAR_STD, CLASSES
from check_device import get_device
from data_loader import get_cifar_data_transforms, get_cifar_data_sets
from models import StagedCIFARNet


device = get_device()

transform_train, transform_test = get_cifar_data_transforms()
_, test_set = get_cifar_data_sets(transform_train, transform_test)


model = StagedCIFARNet().to(device)

# Load each stage's weights
base_path = MODEL_WEIGHTS_PATH.replace('.pth', '')
stage0_weights = torch.load(f"{MODEL_WEIGHTS_PATH}.stage0.pth", map_location=device)
stage1_weights = torch.load(f"{MODEL_WEIGHTS_PATH}.stage1.pth", map_location=device)
stage2_weights = torch.load(f"{MODEL_WEIGHTS_PATH}.stage2.pth", map_location=device)

# Load weights into each stage
model.stage1.load_state_dict(stage0_weights)
model.stage2.load_state_dict(stage1_weights)
model.stage3.load_state_dict(stage2_weights)

model.eval()

# Pick random test image
idx = random.randint(0, len(test_set) - 1)
image, label = test_set[idx]

# Save unnormalized copy
unnorm = image.clone()
for t, m, s in zip(unnorm, CIFAR_MEAN, CIFAR_STD):
    t.mul_(s).add_(m)

np_img = unnorm.permute(1, 2, 0).numpy()

# Run inference
image_batch = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_batch)
    predicted = output.argmax(dim=1).item()

predicted_class = CLASSES[predicted]
true_class = CLASSES[label]


# Print result 
print("------- Inference Result -------")
print(f"Random Test Index: {idx}")
print(f"Predicted class:    {predicted_class}")
print(f"True label:         {true_class}")
print("--------------------------------")


# Display image
plt.figure(figsize=(4, 4))
plt.imshow(np_img)
plt.title(f"Predicted: {predicted_class}\nActual: {true_class}", fontsize=12)
plt.axis("off")
plt.tight_layout()
plt.show()