# Visual Engagement Analysis

In an attempt to predict the correlation between YouTube thumbnails and user engagement, I used the Z-score of view count, normalized by the number of channel subscribers, as the engagement metric. The necessary data was collected via the YouTube API. Initially, the model achieved high accuracy. However, after analyzing the data and examining the distribution of Z-scores, I discovered that the model was overfitting to channel logos, which are often embedded in video thumbnails. As a result, the model primarily learned whether a channel was performing above or below the average, rather than making predictions based on thumbnail content.

To look further into this, I performed a train-test split where images were split by channel name. This revealed the extent of the overfitting, as validation accuracy was only slightly above random chance. Despite the challenge, I found the process both insightful and enjoyable.

> **_Note_**: The dataset, as well as additional collected data via the YouTube API, is not included in this repository.

- **Model:** [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- **Dataset:** [YouTube Thumbnail Dataset](https://www.kaggle.com/datasets/praneshmukhopadhyay/youtube-thumbnail-dataset)