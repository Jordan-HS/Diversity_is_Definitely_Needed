# Diversity is Definitely Needed: Improving Model-Agnostic Zero-shot Classification via Stable Diffusion (CVPRW)  
### [[arXiv]](https://arxiv.org/abs/2302.03298) | [[Datasets]](https://zenodo.org/record/7816671#.ZDUTJo5ByRQ)

Jordan Shipard<sup>1</sup>, Arnold Wiliem<sup>1,2</sup>, Kien Nguyen Thanh<sup>1</sup>, Wei Xiang<sup>3</sup>, Clinton Fookes<sup>1</sup>

[<sup>1</sup>Signal Processing, Artificial Intelligence and Vision Technologies (SAIVT), Queensland University of Technology, Australia](https://research.qut.edu.au/saivt/)  
[<sup>2</sup>Sentient Vision Systems, Australia](https://sentientvision.com/)  
[<sup>3</sup>School of Computing, Engineering and Mathematical Sciences, La Trobe University, Australia](https://www.latrobe.edu.au/school-computing-engineering-and-mathematical-sciences)  

Accepted to the [Generative Models for Computer Vision Workshop](https://generative-vision.github.io/workshop-CVPR-23/) at CVPR 2023

## Model-Agnostic Zero-Shot Classifcation
<img src="https://user-images.githubusercontent.com/41477139/231360240-2bf404a2-3526-40ba-9a67-5d116e66af63.png " data-canonical-src="https://user-images.githubusercontent.com/41477139/231360240-2bf404a2-3526-40ba-9a67-5d116e66af63.png " width="600" height="600" />


| Dataset | Model | Base Class | Best tricks |
| ---     | ---   | ---        | ---         |
|CIFAR10  | CLIP-ResNet50 | 75.6 | - |
|         | ResNet50 | 60.5 | 81 (+20.5)|

## Requirements
For training and testing
* Python 3.8+  
* Pytorch 1.13.0  
* Torchvision 0.14.0

For dataset generation
* 'ldm' environment from [stable-diffusion](https://github.com/CompVis/stable-diffusion)

## Synthetic Datasets
Datasets are hosted on [Zenodo](https://zenodo.org/record/7816671#.ZDYQ145ByRR) with the download links provided in the table below. 
| Dataset | Download |
|---|---|
| CIFAR10 Base Class | [cifar10_generated_32A.tar.gz](https://zenodo.org/record/7816671/files/cifar100_generated_32A.tar.gz?download=1)|
| CIFAR10 Class Prompt | [cifar10_generated_class_prompt_32A.tar.gz](https://zenodo.org/record/7816671/files/cifar10_generated_class_prompt_32A.tar.gz?download=1)|
| CIFAR10 Multi-Domain | [cifar10_generated_multidomain_32A.tar.gz](https://zenodo.org/record/7816671/files/cifar10_generated_multidomain_32A.tar.gz?download=1)|
| CIFAR10 Random Guidance | [cifar10_generated_random_scale_32A.tar.gz](https://zenodo.org/record/7816671/files/cifar10_generated_random_scale_32A.tar.gz?download=1)| 
| CIFAR100 Base Class | [cifar100_generated_32A.tar.gz](https://zenodo.org/record/7816671/files/cifar100_generated_32A.tar.gz?download=1)|
| CIFAR100 Multi-Domain | [cifar100_generated_multidomain_32A.tar.gz](https://zenodo.org/record/7816671/files/cifar100_generated_multidomain_32A.tar.gz?download=1)|
| CIFAR100 Random Scale | [cifar100_generated_random_scale_32A.tar.gz](https://zenodo.org/record/7816671/files/cifar100_generated_random_scale_32A.tar.gz?download=1)
| EuroSAT Base Class | [EuroSat_generated_64.tar.gz](https://zenodo.org/record/7816671/files/EuroSat_generated_64.tar.gz?download=1)|
| EuroSAT Random Scale | [EuroSat_generated_random_scale_64.tar.gz](https://zenodo.org/record/7816671/files/EuroSat_generated_random_scale1_64.tar.gz?download=1)

These are the exact generated synthetic datasets and images used to train the networks in the paper. All datasets were generated using [Stable Diffusion V1.4](https://github.com/CompVis/stable-diffusion). '32A' refers to the image size of 32x32 pixels, which was resized from 512x512 with anti-aliasing. '64' is 64x64 resized from 512x512 without anti-aliasing. Only the datasets which improve performance above the base class (e.g. the best tricks) are currently hosted. If you would like any of the other datasets from the paper either raise an issue, or email me at jordan.shipard@hdr.qut.edu.au.

## Usage
### How to generate your own synthetic datasets

### How to train on the synthetic dataset

### How to test on the real dataset

## Acknowledgements
This work has been supported by the [SmartSat CRC](https://smartsatcrc.com/),
whose activities are funded by the Australian Government’s
CRC Program; and partly supported by [Sentient Vision Systems](https://sentientvision.com/). Sentient Vision Systems is one of the leading Australian developers of computer vision and artificial intelligence software solutions for defence and civilian applications.

## Citation
