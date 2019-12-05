import numpy as np
import matplotlib.pyplot as plt

crops = np.load("crop_200_20100k_40-100.npy")
while True:
    i = np.random.randint(len(crops))
    crop = crops[i]
    plt.imshow(crop[..., ::-1])
    plt.show()
