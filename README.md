# Pytorch implementation of "Sparse Prototype Network for Explainable Pedestrian Behavior Prediction"
<img src="https://github.com/Equinoxxxxx/SPN/blob/main/fig2.png">

## Visualized prototypes
https://github.com/Equinoxxxxx/SPN/tree/main/visualized_prototypes 

## Customize the directories
Change the directories for datasets (```dataset_root```) and weight files (```ckpt_root```) in config.py.

## Prepare the datasets
Download the datasets ([PIE](https://github.com/aras62/PIEPredict?tab=readme-ov-file#PIE_dataset), [TITAN](https://usa.honda-ri.com/titan))  
Note: for PIE, the original data are in .mp4 format, use the scripts from the official repo ([PIE](https://github.com/aras62/PIEPredict?tab=readme-ov-file#PIE_dataset)) to extract the frames.

Extract the data in the following format:
```
[dataset_root]/
    PIE_dataset/
        annotations/
        PIE_clips/
        ...
    JAAD/
        annotations/
        JAAD_clips/
        ...
    TITAN/
        honda_titan_dataset/
            dataset/
                clip_1/
                clip_2/
                ...
    nusc/
        samples/
        v1.0-trainval/
        ...
    BDD100k/
        bdd100k/
            images/track/
                train/
                ...
            labels/box_track_20/
                train/
                ...
```
## Get the weights for Grounded SAM and HRNet
Download the weight files to ```ckpt_root```
[C3D](https://drive.google.com/file/d/19NWziHWh1LgCcHU34geoKwYezAogv9fX/view?usp=sharing)
[HRNet](https://drive.google.com/open?id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS)
[Swin Transformer](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
[ViT](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
```
[ckpt_root]/
    c3d-pretrained.pth
    groundingdino_swint_ogc.pth
    pose_hrnet_w48_384x288.pth
    sam_vit_h_4b8939.pth
```

## Preprocess
Install [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything?tab=readme-ov-file#install-without-docker)
Get cropped images, skeletons and segmentation maps
```
cd SPN
python -m tools.data.preprocess
```

## Train
```
python main.py --test_every 1 --explain_every 10
```
