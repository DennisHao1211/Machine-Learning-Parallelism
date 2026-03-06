

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

MODEL_WEIGHTS_PATH = "cifar_cnn.pth"
CIFAR_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR_STD = (0.24703233, 0.24348505, 0.26158768)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]