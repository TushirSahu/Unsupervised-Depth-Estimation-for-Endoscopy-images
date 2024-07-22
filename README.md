# Unsupervised Synergistic Depth Estimation via Double Decoder CNN for Endoscopic Images
<img src="https://github.com/user-attachments/assets/d20caeee-8da2-4e1c-a49d-ffa81c005d5d" width="1200">

## Setup
We ran our experiments with PyTorch 1.11.0, CUDA 11.3, Python 3.8.13 and Ubuntu 20.04.

## Datasets
You can download the [Endovis or SCARED dataset](https://endovissub2019-scared.grand-challenge.org/) by signing the challenge rules and emailing them to [max.allan@intusurg.com](mailto:max.allan@intusurg.com),  you can download the Hamlyn dataset from this [website](http://hamlyn.doc.ic.ac.uk/vision/).

**Endovis split**

The train/test/validation split for Endovis dataset used in our works is defined in the  `splits/endovis`  folder.

**Data structure**

The directory of dataset structure is shown as follows:
```
/path/to/endovis_data/
  dataset1/
    keyframe1/
      left_img/
          000001.png
```
## Prediction for a single image
```
python test_simple.py --image_path <your_image_or_folder_path> --model_path <depth_model_path> --output_path <path to save results>
```

## Training
You can train a model by running the following command:
```
python train.py --data_path <your_data_path> --log_dir <path_to_save_model>
```
## Evaluation
To prepare the ground truth depth maps run:
```
python export_gt_depth.py --data_path <your_data_path> --split <your_dataset_type>
```
You can evaluate a model by running the following command:
```
python evaluate_depth.py --data_path <your_data_path> --load_weights_folder <your_model_path> --eval_split <your_dataset_type>
```
