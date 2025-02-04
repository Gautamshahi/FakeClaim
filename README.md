# FakeClaim: A Multiple Platform-driven Dataset for Identification of Fake News on 2023 Israel-Hamas War

This GitHub repository corresponds to the dataset used for our research article titled **FakeClaim: A Multiple Platform-driven Dataset for Identification of Fake News on 2023 Israel-Hamas War**.

In our article, we contribute the first publicly available dataset of factual claims from different platforms and fake YouTube videos on the 2023 Israel-Hamas war for automatic fake YouTube video classification. The FakeClaim data is collected from 60 fact-checking organizations in 30 languages and enriched with metadata from the fact-checking organizations curated by trained journalists specialized in fact-checking. Further, we classify fake videos within the subset of YouTube videos using textual information and user comments. We used a pre-trained model to classify each video with different feature combinations. Our best-performing fine-tuned language model, Universal Sentence Encoder (USE), achieves a Macro F1 of 87\%, which shows that the trained model can be helpful for debunking fake videos using the comments from the user discussion. 

## Data 
We have collected the data from existing resources and self-developed a Python program from YouTube. Due to [YouTube Data Sharing Policy](https://www.youtube.com/howyoutubeworks/our-commitments/protecting-user-data/), we are not allowed to share the full video information and comments, but it can be shared based on mutual agreement for research purposes. The data folder contains two files for fake and real videos in the following format.
**videoID** - unique ID of a YouTube video

#### How do I cite this work?

Please cite the [ECIR 2024 paper](https://arxiv.org/abs/2401.16625):
```
@@inproceedings{DBLP:conf/webist/ShahiM24,
  author       = {Gautam Kishore Shahi and
                  Tim A. Majchrzak},
  title        = {Hate Speech Detection Using Cross-Platform Social Media Data in English
                  and German Language},
  booktitle    = {Proceedings of the 20th International Conference on Web Information
                  Systems and Technologies, {WEBIST} 2024, Porto, Portugal, November
                  17-19, 2024},
  pages        = {131--140},
  publisher    = {{SCITEPRESS}},
  year         = {2024},
  url          = {https://doi.org/10.5220/0013070000003825},
  doi          = {10.5220/0013070000003825}
}
```
## Contact information
For help or issues using data, please submit a GitHub issue.

For personal communication related to our work, please contact Gautam Kishore Shahi(`gautamshahi16@gmail.com`)
