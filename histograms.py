import os
import torch
import plotly.graph_objects as go

import configs
from data import ThumbnailDataset

# This script shows you the histogram of the z-scores for each youtube channel

if __name__ == "__main__":
    dataset = ThumbnailDataset(mode="train")
    data: list = dataset.metadata
    mean = dataset.mean
    std_deviation = dataset.std_deviation

    # Get scores for each channel 
    channel_dict = dict()
    for metas in data:
        channel = metas["Channel"]
        views = metas["Views"]
        subscribers = metas["Subscribers"]
        z_score = (views/subscribers - mean)/std_deviation
        channel_dict[channel].append(z_score) if channel in channel_dict.keys() else channel_dict.update({channel: [z_score]})


    # Make histograms
    min_score = min([min(channel_dict[l]) for l in channel_dict.keys()])
    max_score = max([max(channel_dict[l]) for l in channel_dict.keys()])
    bins = torch.arange(start=min_score - 1, end= max_score + 1, step=0.1)
    for ch in channel_dict.keys(): channel_dict[ch] = torch.histogram(torch.tensor(channel_dict[ch]), bins=bins)[0]
    for ch in channel_dict.keys(): channel_dict[ch] = go.Figure().add_trace(go.Bar(x=bins, y=channel_dict[ch])).update_layout(title=ch)
    
    # Render images
    for image in channel_dict.values():
        image.write_image(os.path.join(configs.ROOT, "./preview.jpg"))
        input("Press ENTER for next image!")
