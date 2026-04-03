import torch
import matplotlib.pyplot as plt
import random

from constants import MODEL_WEIGHTS_PATH, CIFAR_MEAN, CIFAR_STD, CLASSES
from check_device import get_device
from data_loader import get_cifar_data_transforms, get_cifar_data_sets
from models import CIFARNet    


device = get_device()

# get our test data set
transform_train, transform_test = get_cifar_data_transforms()
_, test_set = get_cifar_data_sets(transform_train, transform_test)


# load the trained model and put it on the device we are using
model = CIFARNet().to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
model.eval()


# Pick random test image
idx = random.randint(0, len(test_set) - 1)
image, label = test_set[idx]

# Save unnormalized copy
unnorm = image.clone()
for t, m, s in zip(unnorm, CIFAR_MEAN, CIFAR_STD):
    t.mul_(s).add_(m)

np_img = unnorm.permute(1, 2, 0).numpy()

# 4. Run inference first
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


# 5. Display image
plt.figure(figsize=(4, 4))
plt.imshow(np_img)
plt.title(f"Predicted: {predicted_class}\nActual: {true_class}", fontsize=12)
plt.axis("off")
plt.tight_layout()
plt.show()


