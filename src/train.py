# train.py
import torch
import torch.nn as nn
import torch.optim as optim

import time

from constants import LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, EPOCHS, MODEL_WEIGHTS_PATH
from check_device import get_device
from data_loader import cifar10_loaders
from models import CIFARNet


def main():
    device = get_device()
    print("Using device:", device)

    # get our data to use
    train_loader, test_loader = cifar10_loaders()

    # create our model and put it on whatever device we are using, cpu, gpu etc
    model = CIFARNet().to(device)
    # Cross entropy loss is typically used for classification since it effectively computes the difference between 2 probabilities
    loss_function = nn.CrossEntropyLoss()
    # the optimizer is how we take our steps to adjust the weights
    # SGD stands for stochastic gradient descent and it keeps track of momentum to avoid local minima
    # but overall generally follows the negative gradient downwards
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    # automatically changes the learning rate to learn faster at the beginning and then more precise later
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        start_time = time.time()

        # TRAIN LOOP 

        # .train puts our model in training mode which activates features such as dropout
        model.train()
        train_correct = 0
        train_total = 0

        # pull out the images and labels for one batch
        for images, labels in train_loader:
            # we have to move everything to the device we are using
            images, labels = images.to(device), labels.to(device)

            # zeros out all of the stored gradients in our model so they don't accumlate across runs
            optimizer.zero_grad()

            # run the images through the model to obtain our output
            # output is like a set of probabilities for every class  
            # we can take our prediction to be the class with the highest probability
            # this calls the forward method of our model that we implemented
            outputs = model(images)
            # loss measures how far our actual probabilities labels are from the one hot encoded truth labels
            # one hot encoded means the actual labels have a 0 for every class except the correct class
            # for one image the outputs could look something like this [0.1, 0.2, 0.7]
            # with the truth labels looking something like this [0, 0, 1]
            # and then loss would be how far apart those tensors are
            loss = loss_function(outputs, labels)
            # computes the gradients for every parameter in the model wrt to the loss we calculated
            loss.backward()
            # nudges the parameters by the gradient we just calculated to make the model actually learn
            optimizer.step()

            # track training accuracy
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        # update our optimizers learning rate
        scheduler.step()

        train_accuracy = 100 * train_correct / train_total
        print(f"Training Accuracy: {train_accuracy:.2f}%")

        # TEST LOOP 
        # eval turns off things that help with training but hinder performance like dropout
        model.eval()
        correct, total = 0, 0

        # disables gradient calculation so our model can't learn from the testing data and cheat
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Epoch time: {(time.time() - start_time):.2f}")

    # save our weights so we can later use them for inference without having to retrain
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print(f"\nSaved model weights to {MODEL_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()


