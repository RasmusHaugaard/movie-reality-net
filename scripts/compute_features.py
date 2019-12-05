import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision import transforms

import numpy as np

from movie_reality_net.movie_reality_net import MovieRealityNet

tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet mu, std
])

trainset = MovieRealityNet(crop_res=224, transform=tfms)
trainloader = DataLoader(trainset, batch_size=128,
                         shuffle=False, num_workers=10, pin_memory=False)

model = torchvision.models.resnet50(pretrained=True)
model = nn.Sequential(
    *list(model.children())[:-1],
    nn.Flatten(),
)
model.eval()
model.cuda()

F = []
T = []
with torch.no_grad():
    for i, (x, t) in enumerate(trainloader):
        f = model(x.cuda())
        F.append(f.detach().cpu().numpy())
        T.append(t.numpy())
        if i % (len(trainloader) // 20) == 0:
            print(i / len(trainloader))
F = np.vstack(F)
T = np.concatenate(T)

np.save("features.npy", F)
np.save("targets.npy", T)

print("Done")
