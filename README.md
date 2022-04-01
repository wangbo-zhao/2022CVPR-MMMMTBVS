# 2022CVPR-Modeling-Motion-with-Multi-Modal-Features-for-Text-Based-Video-Segmentation
This is the code for CVPR2022 paper "Modeling Motion with Multi-Modal Features for Text-Based Video Segmentation"

🔥🔥🔥Coming Soon🔥🔥🔥
## Usage
1. Download [A2D-Sentences](https://kgavrilyuk.github.io/publication/actor_action/) and [JHMDB-Sentences](https://kgavrilyuk.github.io/publication/actor_action/).

2. Please use [RAFT](https://github.com/princeton-vl/RAFT) to generate the opticla flow map for each frame.

3. Put them as follows:
```
your dataset dir/
└── A2D/ 
    ├── allframes/  
    ├── allframes_flow/
    ├── Annotations_visualize
    ├── a2d_txt
        └──train.txt
        └──test.txt
└── J-HMDB/ 
    ├── allframes/  
    ├── allframes_flow/
    ├── Annotations_visualize
    ├── jhmdb_txt
        └──train.txt
        └──test.txt
```
"Annotations_visualize" contains the GT masks for each target object. We have upload them to [BaiduPan]() for convenience.

## Train

## Inference
 
 
## Citation
Please consider citing our work in your publications if you are interest in our research:
```
wait
```
