import json
import cv2
import numpy as np
from progressbar import progressbar
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, required=True)
parser.add_argument("--unit", default="", choices=["", "k", "M"])
parser.add_argument("--crop_res", type=int, required=True)
parser.add_argument("--min_size", type=float, default=.5)
parser.add_argument("--max_size", type=float, default=1.)

args = parser.parse_args()

movies = json.load(open("movies.json"))  # type: list

crop_size = (args.min_size, args.max_size)  # fac of min(h, w)
crop_res = args.crop_res

N = args.N * {"": 1, "k": int(1e3), "M": int(1e6)}[args.unit]
NM = N // len(movies)
assert N % NM == 0, "N has to be a multiple of len(movies)"

crops = np.empty((N, crop_res, crop_res, 3), dtype=np.uint8)

crop_i = 0
for movie_i, movie in enumerate(movies):
    name = movie["name"]
    print("Movie {}/{}: {}".format(movie_i, len(movies), name))

    cap = cv2.VideoCapture(movie["file_name"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    start, end = movie["start"], movie["end"]
    start_frame, end_frame = start * fps, end * fps

    frames = start_frame + (np.arange(NM) / NM) * (end_frame - start_frame)
    frames = frames.astype(np.int)

    for frame in progressbar(frames):
        #  This is faster when sampling dense frames than setting POS_FRAMES
        while cap.get(cv2.CAP_PROP_POS_FRAMES) != frame:
            _, img = cap.read()

        # crop
        h, w = img.shape[:2]
        size_r = crop_size[0] + np.random.rand() * (crop_size[1] - crop_size[0])
        size_px = int(min(h, w) * size_r)
        top = np.random.randint(0, h - size_px)
        left = np.random.randint(0, w - size_px)
        img = img[top:top + size_px, left:left + size_px]

        # resize
        img = cv2.resize(img, (crop_res, crop_res), interpolation=cv2.INTER_AREA)
        crops[crop_i] = img[..., ::-1]  # bgr -> rgb
        crop_i += 1

name = "crop_{}_{}{}_{}-{}".format(args.crop_res, args.N, args.unit, int(args.min_size * 100), int(args.max_size * 100))
np.save(name, crops)
