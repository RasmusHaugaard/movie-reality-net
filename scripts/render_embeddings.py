import numpy as np
import cv2
import matplotlib.pyplot as plt
from movie_reality_net.movie_reality_net import get_movies_info

movie_info = get_movies_info()
emb_mv_ids = np.load("targets.npy")
frames_per_movie = 100000 // 20

w, h = 1500, 1500
k = 9
mult = 7e-2

img = np.zeros((h, w, 4), np.float)

gauss = cv2.getGaussianKernel(k, k / 5)
gauss = gauss @ gauss.T
gauss /= gauss[k // 2, k // 2]

black = np.zeros((k, k, 4), np.float)
black[:, :, :3] += 0e-1
black[:, :, 3] = gauss * mult
# other colors should be premultiplied

red = black.copy()
red[:, :, 0] = gauss

blue = black.copy()
blue[:, :, 2] = gauss

green = black.copy()
green[:, :, 1] = gauss

mv_is_synth = np.array([m.type == "synth" for m in movie_info])
color_map = {"tangled": green, "the_adventures_of_tintin": blue, "toy_story": red}


def get_color(i):
    movie_id = i // 5000
    color = color_map.get(movie_info[movie_id].name, black)
    return color


emb = np.load("embeddings.npy")
# emb_idx = np.random.randint(0, len(emb), 100000)
# emb = emb[emb_idx]
mi, ma = emb.min(axis=0), emb.max(axis=0)
emb = (emb - mi) / (ma - mi) * (w - k, h - k)

for i, (y, x) in enumerate(emb):
    x, y = int(x), int(y)
    dst = img[y:y + k, x:x + k]
    src = get_color(i)
    src_a, dst_a = src[..., 3], dst[..., 3]
    src_rgb, dst_rgb = src[..., :3], dst[..., :3]
    out_a = src_a + (1 - src_a) * dst_a
    out_rgb = src_rgb + dst_rgb * (1 - src_a.reshape(k, k, 1))
    img[y:y + k, x:x + k, :3] = out_rgb
    img[y:y + k, x:x + k, 3] = out_a

print(img.min(), img.max())
img[img > 1] = 1
plt.imsave("emb_render.png", img)
plt.imshow(img)
plt.show()
