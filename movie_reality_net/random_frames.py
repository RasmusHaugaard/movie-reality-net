import json
import cv2
import numpy as np
import random

movies = json.load(open("movies.json"))  # type: list
random.shuffle(movies)

for movie in movies:
    name = movie["name"]
    start, end = movie["start"], movie["end"]
    cap = cv2.VideoCapture(movie["file_name"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    rand_frames = np.random.randint(start * fps, end * fps, 10)
    for frame in (start * fps, end * fps, *rand_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, img = cap.read()
        cv2.imshow(name, img)
        cv2.waitKey()
    cv2.destroyWindow(name)
