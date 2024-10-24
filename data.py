
import os
import random
import json
import torch
import torchvision

import configs
import utils

from typing import Literal


class ThumbnailDataset(torch.utils.data.Dataset):
    def __init__(self, mode: Literal["train", "test"]):
        metadata_file: str = os.path.join(configs.ROOT, "./thumbnails/metadata.json")
        with open(metadata_file, "r") as f: self.metadata: list = json.load(f)
        self.metadata = [m for m in self.metadata if self._check(m)]
        #random.shuffle(self.metadata)
        if mode == "train": self.data = self.metadata[:int(0.8 * len(self.metadata))]
        if mode == "test": self.data = self.metadata[int(0.8 * len(self.metadata)):]
        self.mean, self.std_deviation = self._metrics(self.data)

    def __getitem__(self, idx: int):
        metas: dict = self.data[idx]
        video_id: str = metas["Id"]
        channel_name: str = metas["Channel"]
        category: str = metas["Category"]
        views: int = metas["Views"]
        subscribers: int = metas["Subscribers"]
        z_score: float = (views/subscribers - self.mean)/self.std_deviation

        # Get image
        channel_dir: str = os.path.join(configs.ROOT, f"./thumbnails/images/{channel_name}")
        image_path: str = os.path.join(channel_dir, f"{video_id}.jpg")
        image: torch.Tensor = torchvision.io.read_image(image_path)
        image = torchvision.transforms.Resize((64, 128))(image)
        return image.float(), z_score.float()
    
    def __len__(self):
        return len(self.data)

    @staticmethod
    def _check(metas):
        video_id: str = metas["Id"]
        channel_name: str = metas["Channel"]
        channel_dir: str = os.path.join(configs.ROOT, f"./thumbnails/images/{channel_name}")
        image_path: str = os.path.join(channel_dir, f"{video_id}.jpg")
        check = bool(f"{video_id}.jpg" in os.listdir(channel_dir))
        if not check: print(f"WARNING: {image_path} does not exist")
        return check
    
    @staticmethod
    def _metrics(data):
        scores = []
        for metas in data:
            views: int = metas["Views"]
            subscribers: int = metas["Subscribers"]
            score: float = views/subscribers
            scores.append(score)
        scores = torch.tensor(scores)
        mean = torch.median(scores)
        sum_of_squared_deviations = torch.sum((scores - mean) ** 2)
        variance = sum_of_squared_deviations/len(scores)
        std_deviation = torch.sqrt(variance)
        return mean, std_deviation


class BlackandWhite(torch.utils.data.Dataset):
    def __init__(self, mode: str):
        self.black = torch.zeros((3,128,128))
        self.white = torch.ones((3,128,128)) * 255

    def __getitem__(self, idx: int):
        randint = random.randint(0, 10)
        if randint > 5: 
            image = self.black
            target = 1
        else:
            image = self.white
            target = -1
        return image.float(), torch.tensor(target)
    
    def __len__(self):
        return 100

if __name__ == "__main__":
    dataset = ThumbnailDataset(mode="test")
    for i in range(len(dataset)):
        img, target = dataset.__getitem__(i)
        print(target)
        utils.save_image(img, os.path.join(configs.ROOT, "./preview.jpg"))
        input("Press ENTER for next image!")