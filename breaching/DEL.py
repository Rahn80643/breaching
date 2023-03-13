import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



train_dataset = datasets.OxfordIIITPet(root='/home/ryhuang/data', split='trainval', transform=transform, download=True)
test_dataset = datasets.OxfordIIITPet(root='/home/ryhuang/data', split='test', transform=transform, download=True)


# Create a grid of 16 images
batch_size = 16
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
images, labels = next(iter(data_loader))
grid = torchvision.utils.make_grid(images, nrow=4)

# Plot the grid of images
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.show()

s =1