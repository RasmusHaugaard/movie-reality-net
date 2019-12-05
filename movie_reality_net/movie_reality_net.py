from PIL import Image
from pathlib import Path
import numpy as np
from torchvision.datasets.vision import VisionDataset
import json


class MovieInfo:
    name: str = None
    file_name: str = None
    start: int = None
    end: int = None
    type: str = None

    @classmethod
    def from_obj(cls, obj):
        mi = cls()
        for a in [a for a in dir(cls) if not a.startswith("__")]:
            if callable(getattr(cls, a)):
                continue
            setattr(mi, a, obj[a])
        return mi

    def to_obj(self):
        obj = {}
        for a in [a for a in dir(self) if not a.startswith("__")]:
            if callable(getattr(self, a)):
                continue
            obj[a] = getattr(self, a)
        return obj

    def __str__(self):
        return json.dumps(self.to_obj(), sort_keys=True, indent=4)


def get_movies_info():
    with open(Path(__file__).parent / "movies.json") as f:
        return [MovieInfo.from_obj(o) for o in json.load(f)]


class MovieRealityNet(VisionDataset):
    def __init__(self, crop_res: int, transform=None):
        super().__init__("~", transform=transform)
        crops_folder = Path(__file__).parent / "crops"

        match = list(crops_folder.glob("crop_{}_*.npy".format(crop_res)))
        assert len(match) == 1

        self.data = np.load(match[0])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is movie index
        """
        img = self.data[index]
        img = Image.fromarray(img)
        target = index // (len(self.data) // 20)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


def main():
    mi = get_movies_info()[0]
    print(mi.name, mi)


if __name__ == '__main__':
    main()
