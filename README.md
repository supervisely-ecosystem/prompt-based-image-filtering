<div align="center" markdown>
<img src=""/>

# Prompt-based image filtering with CLIP

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/СHANGE_THE_NAME!!!!)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/СHANGE_THE_NAME!!!!)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/СHANGE_THE_NAME!!!!)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/СHANGE_THE_NAME!!!!)](https://supervise.ly)

</div>

## Overview

This app allows to fast and easy filtering and sort images in Supervisely datasets by text prompts. It uses [CLIP](https://openai.com/research/clip) model to predict the relevance of images to the given prompt. This app can be useful for filtering or sorting images in a dataset by their content. The model will predict the relevance of each image to the given prompt and show the results in a table along with the score for each image. The user can choose to filter images or sort them by the score or do both at the same time and then upload filtered and/or sorted images to a new dataset.

## How-To-Run

**Step 0:** Run the application from Ecosystem, the context menu of the images project or the images dataset.<br>
Note: if you don't run the app from the context menu of a dataset, first of all, you need to specify the dataset to work with. You need to select a dataset in the `Input dataset` section. After selecting the dataset, click the button `Load data` under the dataset selector. The app will load the dataset and generate a table with all images in the dataset. When the data from the dataset will be loaded, the dataset selector will be locked until you click the `Change dataset` button.<br><br>

**Step 1:** Choose the desired `Model`, and select the `Batch size` (if the default 32 value isn't suitable for your needs). You can uncheck Enable JIT checkbox if you want to use the model without JIT compilation.<br><br>

PLACEHOLDER FOR SCREENSHOT WITH SELECTED DATASET (DATASET THUMBNAIL) AND SELECTED NN MODEL.<br><br>

**Step 2:** Enter the text prompt in the `Text prompt` field. The prompt can be a single word or a phrase. And then click the `Start Inference` button. The app will start with downloading chosen model and then it will start inference of images with specified batch size. You can stop the inference process at any time by clicking the `Cancel inference` button.<br><br>

**Step 3:** After the inference is finished, the next section of the app will be unlocked. In this section, you can see a chart that visualizes the number of images and score range. You can also see a table with all images from the dataset and their scores, which are sorted by score (confidence) in descending order by default. You can press the `Select` button in the table to preview the image. It can be handy for finding the optimal threshold for image filtering.<br><br>

PLACEHOLDER FOR SCREENSHOT WITH CHART, TABLE AND ANY IMAGE IN PREVIEW (NOT PLACEHOLDER)<br><br>

**Step 4:** In the final step you need to select the output dataset for images (note, that the same dataset can not be selected) and define how the app should handle images in the dataset: filter, sort or both. Note, that at least one of the following options should be selected:<br>

- `Filter images` - the app will filter images by the score threshold and upload only images whose score is higher or lower than the selected `Threshold`. You can choose which images should be kept: above or below the threshold.<br>
- `Sort images` - the app will sort images by the score and upload them to the output dataset in descending order or ascending order.<br>

You can also check `Add confidence tag` checkbox to add a tag with the confidence score to each image in the output dataset. Note, that this option can slow down the app.<br><br>

PLACEHOLDER FOR SCREENSHOT WITH RESULTS<br><br>

After the upload is finished, you will see a message with the number of images that have been successfully uploaded to the dataset. The app will also show the project and the dataset to which the images were uploaded. You can click on the links to open the project or the dataset.<br><br>
After finishing using the app, don't forget to stop the app session manually in the App Sessions. The app will write information about the text prompt and CLIP score to the image metadata. You can find this information in the Image Properties - Info section of the image in the labeling tool.
