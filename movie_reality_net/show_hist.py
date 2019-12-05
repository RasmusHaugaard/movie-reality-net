import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

movies = json.load(open("movies.json"))  # type: list

crop_size = 0.5  # of min(h, w)
crop_res = 224

imgs = []
var_scores = []
edge_scores = []

for _ in range(100):
    movie = np.random.choice(movies)
    name = movie["name"]
    start, end = movie["start"], movie["end"]
    cap = cv2.VideoCapture(movie["file_name"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    rand_frame = np.random.randint(start * fps, end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, rand_frame)
    _, img = cap.read()

    # crop
    h, w = img.shape[:2]
    s = int(min(h, w) * crop_size)
    top = np.random.randint(0, h - s)
    left = np.random.randint(0, w - s)
    img = img[top:top + s, left:left + s].astype(float) / 255

    # scores for edges
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=9)
    edge_score = np.abs(lap).mean()

    # variation score
    pixels = img.reshape(-1, 3)
    var_score = np.max(pixels.max(axis=0) - pixels.min(axis=0))

    if False:
        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        axs[0, 0].imshow(img[..., ::-1])
        axs[0, 0].set_title("var score: {:2f}".format(var_score))
        axs[0, 1].imshow(lap[..., ::-1] * 2e-2 + 0.5, cmap="gray")
        axs[0, 1].set_title("edge_score: {:2f}".format(edge_score))
        axs[0, 2].axis("off")

        for ci, ch, cn in zip(range(3), img[..., ::-1].transpose((2, 0, 1)), "rgb"):
            axs[1, ci].imshow(ch, cmap="gray")
            axs[1, ci].set_title(cn)
            axs[2, ci].hist(ch.reshape(-1))
            axs[2, ci].set_xlim(0, 1)

        plt.show()

    imgs.append(img)
    edge_scores.append(edge_score)
    var_scores.append(var_score)

fig, axs = plt.subplots(1, 3)
axs[0].hist(edge_scores)
axs[0].set_title("edge_scores")
axs[1].hist(var_scores)
axs[1].set_title("var_scores")
axs[2].scatter(edge_scores, var_scores)
axs[2].set_xlabel("edge score")
axs[2].set_ylabel("var score")


def on_click(e):
    if e.inaxes is not axs[2]:
        return
    datapoints = np.array([edge_scores, var_scores]).T
    dists = np.linalg.norm(datapoints - (e.xdata, e.ydata), axis=-1)
    i = np.argmin(dists)
    img = imgs[i]
    fig = plt.figure()
    plt.imshow(img[..., ::-1])
    fig.show()


fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()