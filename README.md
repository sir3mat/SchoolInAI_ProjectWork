# School in AI Project Work

<br />
<div align="center">
  <a href="https://aischools.it/">
    <img src="http://www.aiacademy.unimore.it/media/news/ai-logo-white_2ND_EDITION.png" alt="Logo" >
  </a>
  <h1 align="center">Domain adaption on pedestrian detection with Faster R-CNN</h3>
  <h3 align="center">
    This repository contains the code of the "School in AI: Deep Learning, Vision and Language for Industry - second edition" final project work by Matteo Sirri.
  </p>
</div>


## Demo Links

|                                                                    Google Colab Demo                                                                    |                                                                       Huggingface Demo                                                                        |                                                    Report                                                     |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
| [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" width="500"/>](https://colab.research.google.com/drive/1KQqmPANWiLqAJH0yZN1UV_FVqnzPrurw) | [<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces" width="620"/>](https://huggingface.co/spaces/sir3mat/SchoolInAiProjectWork) | [<img src="https://img.shields.io/badge/Report-Download-Blue?style=flat&color=informational" width="500"/>](https://drive.google.com/file/d/1mVug8X3YX9Yy2A383XlWHXy30VMS8f4v/view?usp=share_link) |
- Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio).

## Installation

N.B.: Installation only avaiable in win64 environments

Create and activate an environment with all required packages:

```
conda create --name pedestrian_detector --file deps/wins/conda_requirements.txt
conda activate pedestrian_detector
pip install -r deps/win/pip_requirements.txt
```

## Dataset download and preparation:

### Prepare MOTSynth dataset

1. Download MOTSynth_1.

```
wget -P ./storage/MOTSynth https://motchallenge.net/data/MOTSynth_1.zip
unzip ./storage/MOTSynth/MOTSynth_1.zip -d ./storage/MOTSynth/
rm ./storage/MOTSynth/MOTSynth_1.zip
```

2. Delete video from 123 to 256
3. Extract frames from the videos

```
python tools/anns/to_frames.py --motsynth-root ./storage/MOTSynth

# now you can delete other videos
rm -r ./storage/MOTSynth/MOTSynth_1
```

4. Download and extract annotations

```
wget -P ./storage/MOTSynth https://motchallenge.net/data/MOTSynth_coco_annotations.zip
unzip ./storage/MOTSynth/MOTSynth_coco_annotations.zip -d ./storage/MOTSynth/
rm ./storage/MOTSynth/MOTSynth_coco_annotations.zip
```

5. Prepare combined annotations for MOTSynth from the original COCO annotations

```
python tools/anns/combine_anns.py --motsynth-path ./storage/MOTSynth
```

6. Prepare motsynth ouput dir for training results

```
mkdir ./storage/motsynth_output
```

### Prepare MOT17 dataset

1. Download MOT17

```
wget -P ./storage/MOTChallenge https://motchallenge.net/data/MOT17.zip
unzip ./storage/MOTChallenge/MOT17.zip -d ./storage/MOTChallenge
rm ./storage/MOTChallenge/MOTSynth_1.zip
```

2. Generate COCO format annotations

```
python tools/anns/motcha_to_coco.py --data-root ./storage/MOTChallenge
```

### Download pretrained models folder from GDrive

You can find all pretrained models here https://drive.google.com/drive/folders/15Lv40x3MquSnKbI4U5aGSZtqQuEmiwMH?usp=share_link (download them and paste the .pth files in storage/pretrained_models directory).

### Storage directory tree

After runnning this step, your storage directory should look like this:

```text
storage
    ├── MOTChallenge
        ├── MOT17
        ├── motcha_coco_annotations
    ├── MOTSynth
        ├── annotations
        ├── comb_annotations
        ├── frames
    ├── motsynth_output
    ├── pretrained_models
```

## Object Detection

An adaption of torchvision's detection reference code is done to train Faster R-CNN on a portion of the MOTSynth dataset.

- To train the model you can run (change params in the script):

```
./scripts/train_detector.sh
```

- To fine-tuning the model you can run (change params in the script):

```
./scripts/fine_tuning_detector.sh
```

- To evaluate the model you can run (change params in the script):

```
./scripts/evaluate_detector.sh
```

- To make inference and show results you can run (change params in the script):

```
./scripts/inference_detector.sh
```

## Contact

Matteo Sirri - matteosirri99@gmail.com

