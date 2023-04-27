<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/115161827/233118282-30d61976-a80e-46cf-be93-ca3388c816fb.jpg"/>

# Prompt-based Image Filtering with CLIP

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Pretrained-models">Pretrained models</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/prompt-based-image-filtering)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/prompt-based-image-filtering)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/prompt-based-image-filtering)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/prompt-based-image-filtering)](https://supervise.ly)

</div>

# Overview

This app allows you to quickly and easily filter and rank images in Supervisely datasets by text prompts. It uses [CLIP](https://openai.com/research/clip) model to predict the **relevance** of images to the given text prompt. This app can be useful for filtering or ranking images in a dataset by their content. The relevance (CLIP score) of each image to the given prompt will be shown in a table. The user can choose to filter or sort images by relevance or do both at the same time and then upload images to a new dataset.

# Pretrained models

We have selected several pre-trained models from the [OpenCLIP](https://github.com/mlfoundations/open_clip) repository:

| Model                | Pretrained on                      | top-1 accuracy on ImageNet | Size    |
| -------------------- | ---------------------------------- | -------------------------- | ------- |
| coca_ViT-L-14        | mscoco_finetuned_laion2B-s13B-b90k | -                          | 2.55 GB |
| coca_ViT-L-14        | laion2B-s13B-b90k                  | 75.5%                      | 2.55 GB |
| ViT-L-14             | openai                             | 75.5%                      | 933 MB  |
| ViT-L-14             | laion2b_s32b_b82k                  | 75.3%                      | 933 MB  |
| ViT-L-14-336         | openai                             | -                          | 933 MB  |
| ViT-g-14             | laion2b_s34b_b88k                  | 78.5%                      | 5.47 GB |
| ViT-bigG-14          | laion2b_s39b_b160k                 | 80.1%                      | 10.2 GB |
| convnext_base_w      | laion2b_s13b_b82k_augreg           | 71.5%                      | 718 MB  |
| convnext_large_d_320 | laion2b_s29b_b131k_ft_soup         | 76.9%                      | 1.41 GB |

# How To Run

**Step 0:** Run the application from Ecosystem, the context menu of the images project or the images dataset.<br>
Note: if you don't run the app from the context menu of a dataset, first of all, you need to specify the dataset to work with. You need to select a dataset in the `Input dataset` section. After selecting the dataset, click the button `Load data` under the dataset selector. The app will load the dataset and generate a table with all images in the dataset. When the data from the dataset will be loaded, the dataset selector will be locked until you click the `Change dataset` button.<br><br>

**Step 1:** Choose the desired `Model`, and select the `Batch size` (if the default 32 value isn't suitable for your needs). You can uncheck Enable JIT checkbox if you want to use the model without JIT compilation.<br><br>

<img src="https://user-images.githubusercontent.com/115161827/232123410-239309d8-e65a-492e-8617-427424359660.png" />
<br><br>

**Step 2:** Enter the text prompt in the `Text prompt` field. The prompt can be a single word or a phrase. And then click the `Start Inference` button. The app will start with downloading chosen model and then it will start inference of images with specified batch size. You can stop the inference process at any time by clicking the `Cancel inference` button.<br><br>

<img src="https://user-images.githubusercontent.com/115161827/234807371-d21ce284-0796-4825-ab75-6f4d86d8bd46.png" />
<br><br>

**Step 3:** After the inference is finished, the next section of the app will be unlocked. The chart shows a CLIP's score (on Y-axis) for each image (X-axis is for image indices), and the images are sorted by the scores in descending order. This can give you an intuition of what kind of data you have in general (e.g. the number of images within a score range) and help to select a threshold. You can also see a table with all images from the dataset and their scores, which are sorted by score in descending order by default. You can press the `Select` button in the table to preview the image. It can be handy for finding the optimal threshold for image filtering.<br><br>

<img src="https://user-images.githubusercontent.com/115161827/232123378-49a885c7-7656-4ec1-85f2-f8d3be5d3597.png" /> <br><br>

**Step 4:** In the final step you need to select the output dataset for images (note, that the same dataset can not be selected) and define how the app should handle images in the dataset: filter, sort or both. Note, that at least one of the following options should be selected:<br>

- `Filter images` - the app will filter images by the score threshold and upload only images whose score is higher or lower than the selected `Threshold`. You can choose which images should be kept: above or below the threshold.<br>
- `Sort images` - the app will sort images by the score and upload them to the output dataset in descending order or ascending order.<br>

You can also check `Add confidence tag` checkbox to add a tag with the confidence score to each image in the output dataset. Note that this option can slow down uploading images.<br><br>

![screen-clip](https://user-images.githubusercontent.com/115161827/233337650-e19f35b9-b537-4ee3-926e-57b0bd074f36.png) <br><br>

After the upload is finished, you will see a message with the number of images that have been successfully uploaded to the dataset. The app will also show the project and the dataset to which the images were uploaded. You can click on the links to open the project or the dataset.<br><br>
After finishing using the app, don't forget to stop the app session manually in the App Sessions. The app will write information about the text prompt and CLIP score to the image metadata. You can find this information in the Image Properties - Info section of the image in the labeling tool.

# Acknowledgment

This app is based on the great work `CLIP`: 

- [GitHub](https://github.com/openai/CLIP) ![GitHub Org's stars](https://img.shields.io/github/stars/openai/CLIP?style=social)

- [Pre-trained models from OpenCLIP](https://github.com/mlfoundations/open_clip)
