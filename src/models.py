# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# all networks must extend the nn.Module
class CIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        # networks must define a constructor to create all of the layers that the network will use

        # our network has 4 convolution layers with 3 mlp layers
        # Our original images have a size of 3x32x32 (3 for rgb then 32x32 image sizes)
        # so we can view each image as having 3 "channels" i.e 3 seperate images
        # and we convolute over that with 64 different kernels to create 64 "new" images
        # the convulution matrix will look over all input channels at once so it will take all rgb into account in one go
        # we then create several more convolution layers increasing the number of channels each time
        # kernel_size 3 means we convolute our images with a 3x3 matrix, padding 1 means we make our image 1 pixel bigger on each side
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # max pool is an operation that just takes the max from a 2x2 grid
        # it is used to reduce the size of images for faster processing without missing key details
        self.pool = nn.MaxPool2d(2, 2)

        # Batch norm normalizes an entire input to the layer of a neural network to keep all activations reasonable
        # we batch normalize for each convolution layer
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)


        # dropout means to randomly turn off half the nodes to prevent overfitting
        # this is only used during the training phase and not during testing
        self.dropout = nn.Dropout(p=0.5)

        # the final step in our network is the full connected layers of which we have 3 of
        # We have as input to the first linear layer 512 * 8 * 8 nodes since the last layer of convolution
        # has dimention (B, 512, 8, 8) so that allows one node for each pixel/channel
        # the last layer has 10 nodes since there are 10 different classes we want to identify
        self.fc1 = nn.Linear(512 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # the forward method defines how data is passed through the layers we created in init

        # for our convolution layers we follow the same basic pattern
        # convlute the input tensor (the raw image)
        # max pool it to reduce the size
        # batch normalize
        # relu (activation function) max(0, x)
        # starting data size is (B, 3, 32, 32)
        x = self.conv1(x) # (B, 64, 32, 32) is the output where B is batch size
        x = self.pool(x) # (B, 64, 16, 16) output from pool, notice the size of our data is cut in 1/4
        x = self.bn1(x) # batch normalization doesn't change the size of the data just normalizes the activations
        x = F.relu(x) # (B, 64, 16, 16) doesn't change the size of the data

        x = self.conv2(x) # (B, 128, 16, 16)
        x = self.pool(x) # (B, 128, 8, 8) notice size of data cut in 1/4 again
        x = self.bn2(x)
        x = F.relu(x) # (B, 64, 8, 8)

        x = self.conv3(x) # (B, 256, 8, 8)
        # we don't pool here since our image is already now pretty small
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x) # (B, 512, 8, 8)
        x = self.bn4(x)
        x = F.relu(x)

        # final shape of data after all convolutions (B, 512, 8, 8)


        # to prepare to put our tensore in the neural network we must flatten it
        # the 1 means to start flattening with the first dimension so only the 512, 8, 8 are flattened
        # this is critical because without that we would flatten all batches into one giant tensore which would be useless
        x = torch.flatten(x, 1) # (B, 512 * 8 * 8) = (B, 32768) output of flatten

        # now we can pass through our flattened tensore into the fully connected layer and another relu activation
        x = self.fc1(x) # (B, 512)
        x = F.relu(x)

        # randomly turn some of the values to 0 to prevent overfitting
        x = self.dropout(x)
        x = self.fc2(x) # (B, 256)
        x = F.relu(x)

        x = self.dropout(x)
        # pass through last fc layer but do not relu because we will softmax later to get probabilities for each class
        x = self.fc3(x) # (B, 10) final size of output

        return x
    

class StagedCIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(1)
        )
        
        self.stage3 = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    layers = [
        ("Input\n3×32×32", 1, 5),
        ("Conv+Pool\n+BN+ReLU\n64×16×16", 2.8, 5),
        ("Conv+Pool\n+BN+ReLU\n128×8×8", 4.6, 5),
        ("Conv+BN\n+ReLU\n256×8×8", 6.4, 5),
        ("Conv+BN\n+ReLU\n512×8×8", 8.2, 5),
        ("Flatten\n32768", 10, 5),
        ("FC+ReLU\n+Dropout\n512", 11.8, 5),
        ("FC+ReLU\n+Dropout\n256", 13.6, 5),
        ("FC\n(Output)\n10 classes", 15.4, 5)
    ]
    
    for i, (label, x, y) in enumerate(layers):
        box = FancyBboxPatch((x-0.6, y-0.8), 1.2, 1.6, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='black', facecolor='lightblue', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, weight='bold')
        
        if i < len(layers) - 1:
            arrow = FancyArrowPatch((x+0.6, y), (layers[i+1][1]-0.6, layers[i+1][2]),
                                   arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
            ax.add_patch(arrow)
    
    plt.title('CIFARNet Architecture', fontsize=18, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('cifar_net_simple.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Simple visualization saved as cifar_net_simple.png")


if __name__ == "__main__":
    main()

