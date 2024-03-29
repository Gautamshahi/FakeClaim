# FakeClaim: A Multiple Platform-driven Dataset for Identification of Fake News on 2023 Israel-Hamas War

This GitHub repository corresponds to the dataset used for our research article titled **FakeClaim: A Multiple Platform-driven Dataset for Identification of Fake News on 2023 Israel-Hamas War**.

In our article, we contribute the first publicly available dataset of factual claims from different platforms and fake YouTube videos on the 2023 Israel-Hamas war for automatic fake YouTube video classification. The FakeClaim data is collected from 60 fact-checking organizations in 30 languages and enriched with metadata from the fact-checking organizations curated by trained journalists specialized in fact-checking. Further, we classify fake videos within the subset of YouTube videos using textual information and user comments. We used a pre-trained model to classify each video with different feature combinations. Our best-performing fine-tuned language model, Universal Sentence Encoder (USE), achieves a Macro F1 of 87\%, which shows that the trained model can be helpful for debunking fake videos using the comments from the user discussion. 

## Data 
We have collected the data using [AAMUSED](https://doi.org/10.1007/978-3-031-10525-8_23). Due to [YouTube Data Sharing Policy](https://www.youtube.com/howyoutubeworks/our-commitments/protecting-user-data/), we are not allowed to share the full video information and comments, but it can be shared based on mutual agreement for research purposes. The data folder contains two files for fake and real videos in the following format.
**videoID** - unique ID of a YouTube video

#### How do I cite this work?

Please cite the [ECIR 2024 paper](https://arxiv.org/abs/2401.16625):
```
@InProceedings{shahiecir2024,
author="Shahi, Gautam Kishore and Jaiswal, Amit Kumar and Mandl, Thomas",
title="FakeClaim: A Multiple Platform-Driven Dataset for Identification of Fake News on 2023 Israel-Hamas War",
booktitle="Advances in Information Retrieval",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="66--74"
}
```
## Contact information
For help or issues using data, please submit a GitHub issue.

For personal communication related to our work, please contact Gautam Kishore Shahi(`gautamshahi16@gmail.com`)
## More update
For more updates on the related publication on the topic of FakeCovid, please visit [WarClaim: 2023-Israel-Hamas-war Dataset](https://github.com/Gautamshahi/WarClaim/) 
