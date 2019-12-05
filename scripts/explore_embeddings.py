import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, KeyEvent
import numpy as np
from scipy.spatial import KDTree

from movie_reality_net.movie_reality_net import get_movies_info, MovieRealityNet

movies_info = get_movies_info()

embeddings = np.load("embeddings.npy")
img_mv_ids = np.load("targets.npy")
mv_idx_is_synth = np.array([mi.type == "synth" for mi in movies_info])
img_idx_is_synth = mv_idx_is_synth[img_mv_ids]
synth_real_scatter_colors = np.array([
    [1., 0., 0., .01],  # real
    [0., 0., 1., .01],  # synth
])
img_scatter_colors = synth_real_scatter_colors[img_idx_is_synth.astype(int)]

imgs = MovieRealityNet(224).data


def split_emb(emb):
    return {"x": emb[:, 0], 'y': emb[:, 1]}


fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.scatter(**split_emb(embeddings), c=img_scatter_colors, edgecolor='none')
ax.axis('off')

kd_emb = KDTree(embeddings)

click_fig, click_ax = plt.subplots(1, 1, figsize=(5, 5))
hover_fig, hover_ax = plt.subplots(1, 1, figsize=(5, 5))
plt.tight_layout(pad=0)


def get_img(e):
    d, i = kd_emb.query((e.xdata, e.ydata))
    return imgs[i]


def on_click(e):
    if e.inaxes is not ax:
        return
    click_ax.clear()
    click_ax.imshow(get_img(e)[..., ::-1])
    click_ax.axis('off')
    click_fig.canvas.draw()


def on_hover(e):
    if e.inaxes is not ax:
        return
    hover_ax.clear()
    hover_ax.imshow(get_img(e)[..., ::-1])
    hover_ax.axis('off')
    hover_fig.canvas.draw()


def on_key_press(e: KeyEvent):
    if e.key == "right":
        on_key_press.i += 1
    elif e.key == "left":
        on_key_press.i -= 1
    else:
        return
    on_key_press.i %= len(movies_info) + 1
    ax.clear()
    if on_key_press.i == len(movies_info):  # synth / real
        mask = mv_idx_is_synth[img_mv_ids]
        scatter_colors = synth_real_scatter_colors
        ax.set_title("synth / real")
    else:
        mask = img_mv_ids == on_key_press.i
        scatter_colors = np.array([
            [1., 0., 0., .01],  # others
            [0., 0., 1., .05],  # current
        ])
        ax.set_title(movies_info[on_key_press.i].name)
    img_scatter_colors = scatter_colors[mask.astype(int)]
    ax.scatter(**split_emb(embeddings), c=img_scatter_colors, edgecolor='none')
    ax.axis('off')
    fig.canvas.draw()


on_key_press.i = len(movies_info)

fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("motion_notify_event", on_hover)
fig.canvas.mpl_connect("key_press_event", on_key_press)
plt.show()
