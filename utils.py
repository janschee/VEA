import requests
import csv
import json
import configs
import torch
import torchvision

def save_image(image: torch.TensorType, path: str) -> None: 
    torchvision.utils.save_image(image.float()/255, path)

def get_video_data(video_id: str):
    api_key: str = configs.API_KEY

    #Get view count
    video_url: str = f"https://www.googleapis.com/youtube/v3/videos?part=statistics,snippet&id={video_id}&key={api_key}"
    response = requests.get(video_url)
    if response.status_code == 200: 
        video_data = response.json()
        view_count = video_data['items'][0]['statistics']['viewCount']
        channel_id = video_data['items'][0]['snippet']['channelId']
    else: 
        print(f'Error: {response.status_code}, {response.text}')
        raise AssertionError

    #Get number of subscribers
    channel_url: str = f"https://www.googleapis.com/youtube/v3/channels?part=statistics&id={channel_id}&key={api_key}"
    response = requests.get(channel_url)
    if response.status_code == 200:
        channel_data = response.json()
        subscriber_count = channel_data['items'][0]['statistics'].get('subscriberCount', 'Not available')
    else: 
        print(f'Error: {response.status_code}, {response.text}')
        raise AssertionError

    return {"ChannelId": channel_id, "Views": int(view_count), "Subscribers": int(subscriber_count)}

def download_video_data():
    # Request data
    json_data: list = []
    with open(configs.METADATA_CSV, mode="r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            print(f"INFO: Processing sample {i}")
            video_id: str = row["Id"]
            try: video_data: dict = get_video_data(video_id)
            except: continue
            channel_id, views, subscribers = video_data["ChannelId"], video_data["Views"], video_data["Subscribers"]
            row["ChannelId"] = channel_id
            row["Views"] = views
            row["Subscribers"] = subscribers
            json_data.append(row)

            # Save data
            with open(configs.METADATA_JSON, mode="w") as f: json.dump(json_data, f, indent=4)

