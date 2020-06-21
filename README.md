# Pytorch implementation for Stylization-Based Architecture for Fast Deep Exemplar Colorization
The project is not just for ‘Stylization-Based Architecture for Fast Deep Exemplar Colorization’. We have also improved the colorization network for better visual results or different usages. We will continue to do the experiments for new ideas, organize the code and  upload the weight files for people who is interested in it. Welcome  to join us to maintain the project together.

## Install Dependencies
The code is written in Python 3.5 using the main following libraries:
```
python >=3.5,PyTorch>=0.4
Requirements: opencv-python,tensorboardX,visdom
Platforms: Ubuntu16.04,cuda-9.0  
```
	
## Data
Following the paper, training: download the coco dataset for transfer sub-net and he ImageNet dataset for colorization sub-net respectively. 
The test images in the paper comes from other colorization tasks or style
transfer projects.

## Architecture  
Follow the folder structure given below. 
```
├── dataset
│   └── Coco
│   └── Imagenet
├── checkpoints
│   └── 02_22_13_48
│   └── 02_25_15_33
│   └── siggraph_latest_net_G.pth
│   └── update_siggraph.pth
├── logs
├── options
│   └──base_options.py
│   └──train_options.py
├── models
│   └── network.py
│   └── RDBN.py
│   └── siggraph.py
│   └── siggraph_sample.py
├── transfer_subnet
│   └── consistencyChecker
│   └── checkpoints
│   └── video_checkpoints
│   └── segmentation
│   		└── ...
│   		└── ...
│   └── utils
│   		└── core.py
│   		└── io.py
│   		└── photo_adin.py
│   └── outputs
│   └── ade20k_semantic_rel.npy
│   └── compare_model.py
│   └── video_dataset.py
│   └── dataset.py
│   └── utilities.py
│   └── flowlib.py
│   └── make_consistencyChecker_script.py
│   └── make_video2image_script.py
│   └── wrap_xiaoke.py
│   └── xiaokemodel.py
│   └── xiaoketransfer.py
│   └── xiaoketransfer2.py
├── util
│   ├── get_data.py
│   ├── html.py
│   ├── image_pool.py
│   ├── util.py
│   └── visualizer.py
├── train.py
├── test.py
├── README.md

```

## Video
We modified the transfer sub-net and transfer the style(artistic style, photo realistic style) on the image to the video by using optical flow to solve the consistency problem.

## Contact
If you find any problems , please feel free to contact me (936214756@qq.com). A brief self-introduction is required.

## Acknowledgments
Our code architecture is inspired by [richzhang](https://richzhang.github.io/ideepcolor/)