from pathlib import Path
from itertools import chain
import json

root = Path()
folders = [root / n for n in ("real", "synth")]
movies = []


def get_seconds(t: str):
    hh, mm, ss = map(int, t.split(":"))
    return hh * 3600 + mm * 60 + ss

# get limits from csv
limits = {}
for line in open(root / "movie_limits.csv").readlines():
    tkns = line.strip().split(",")
    if len(tkns) == 1:
        continue
    assert len(tkns) == 3, "can't parse: " + line
    name, start, end = tkns
    start, end = map(get_seconds, (start, end))
    limits[name] = (start, end)

# find movies in folders
for folder in folders:
    f_movies = chain(folder.glob("*.mp4"), folder.glob("*.mkv"))
    for movie in f_movies:
        file_name = movie.name
        name = movie.name.split(".")[0]
        start = limits[name][0]
        end = limits[name][1]
        movies.append({
            "name": name,
            "file_name": str(movie),
            "start": start,
            "end": end,
            "type": folder.name,
        })

movies.sort(key=lambda m: m["name"])
json.dump(movies, open("movies.json", "w"), indent=4)
