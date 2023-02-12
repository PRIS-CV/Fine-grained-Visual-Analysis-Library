# Fine-grained-Visual-Analysis-Library

## Introduction

FGVCLib is an open-source and well documented library for Fine-grained Visual Classification. It is based on Pytorch with performance and friendly API. Our code is pythonic, and the design is consistent with torchvision. You can easily develop new algorithms, or readily apply existing algorithms.
The branch works with **torch 1.12.1**, **torchvision 0.13.1**.

For more details and the tutorials about the FGVCLib, see [FGVCLib](https://pris-cv-fgvclib.readthedocs.io/en/latest/index.html)

<details open>
<summary>Major features</summary>

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **State of the art**
  We implement state-of-the-art methods by the FGVCLib, [PMG](https://arxiv.org/abs/2003.03836v3), [PMG_V2](https://ieeexplore.ieee.org/abstract/document/9609669), [MCL](https://arxiv.org/abs/2002.04264), [API-Net](https://arxiv.org/abs/2002.10191), [CAL](https://ieeexplore.ieee.org/document/9710619), [TransFG](https://ieeexplore.ieee.org/document/9710619), [PIM](https://arxiv.org/abs/2202.03822). 


## Installation

Please refer to [Installation](https://pris-cv-fgvclib.readthedocs.io/en/latest/get_started.html) for installation instructions.

## Getting started 

Please see [get_started.md](https://pris-cv-fgvclib.readthedocs.io/en/latest/get_started.html) for the basic usage of FGVCLib. We provide the tutorials for:

- [with existing data existing model](https://pris-cv-fgvclib.readthedocs.io/en/latest/1_exist_data_model.html)
- [with existing data new model](https://pris-cv-fgvclib.readthedocs.io/en/latest/2_exist_data_new_model.html)
- [learn about apis](https://pris-cv-fgvclib.readthedocs.io/en/latest/tutorials/tutorial1_apis.html)
- [learn about configs](https://pris-cv-fgvclib.readthedocs.io/en/latest/tutorials/tutorial2_configs.html)
- [learn about criterions](https://pris-cv-fgvclib.readthedocs.io/en/latest/tutorials/tutorial3_criterions.html)
- [learn about datasets](https://pris-cv-fgvclib.readthedocs.io/en/latest/tutorials/tutorial4_datasets.html)
- [learn about metrics](https://pris-cv-fgvclib.readthedocs.io/en/latest/tutorials/tutorial5_metrics.html)
- [learn about model](https://pris-cv-fgvclib.readthedocs.io/en/latest/tutorials/tutorial6_model.html)
- [learn about transforms](https://pris-cv-fgvclib.readthedocs.io/en/latest/tutorials/tutorial7_transform.html)
- [learn about the tools](https://pris-cv-fgvclib.readthedocs.io/en/latest/useful_tools.html)


</details>

## Overview of Benchmark and Model Zoo

<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Fine-grained Visual Classification</b>
      </td>
      <td>
        <b>Other</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/resnet">Baseline_ResNet50</a></li>
            <li><a href="configs/mutual_channel_loss">Mutual-Channel-Loss</a></li>
            <li><a href="configs/progressive_multi_granularity_learning">PMG-ResNet50</a></li>
            <li><a href="configs/progressive_multi_granularity_learning">PMG_V2_ResNet50</a></li>
            <li><a href="configs/">API-Net</a></li>
            <li><a href="configs/">CAL</a></li>
            <li><a href="configs/">TransFG</a></li>
            <li><a href="configs/">PIM</a></li>
      </ul>  
      </td>
      <td>
        <ul>
            <li>visualization</li>
      </ul>  
      </td>
    </tr>
  </tbody>
</table>

<div align="center">
  <b>Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Encoders</b>
      </td>
      <td>
        <b>Heads</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Sotas</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li>Resnet</li>
            <li>VGG</li>
      </ul>  
      </td>
      <td>
        <ul>
            <li>Global Max Pooling</li>
            <li>Global Avg Pooling</li>
            <li>Max Pooling 2d</li>
      </ul>  
      </td>
      <td>
        <ul>
            <li>Classifier_1_FC</li>
            <li>Classifier_2_FC</li>
      </ul>  
      </td>
      <td>
        <ul>
            <li>Multi-scale Convolution neck</li>
      </ul>  
      </td>
      <td>
        <ul>
            <li>Baseline_ResNet50</li>
            <li><a href="configs/mutual_channel_loss/README.md">Mutual-Channel-Loss</a></li>
            <li><a href="configs/progressive_multi_granularity_learning/README.md">PMG-ResNet50</a></li>
            <li>PMG_V2_ResNet50</li>
            <li><a href="configs/">API-Net</a></li>
            <li><a href="configs/">CAL</a></li>
            <li><a href="configs/">TransFG</a></li>
            <li><a href="configs/">PIM</a></li>
      </ul>  
      </td>
    </tr>
  </tbody>
</table>

<!-- ## The Result of the SOTA
We used fgvclib to replicate the state-of-the-art model, and the following table shows the results of our experiment. 

| SOTA       | Result of the paper | Result of the official code | Result of the FGVCLib |
| ---------- | ------------------- | --------------------------- | --------------------- |
| API-Net    |        88.1         |            87.2             |         86.8          |
| CAL        |        90.6         |            89.6             |         89.5          |
| TransFG    |        91.7         |            91.1             |         89.3          |
| PIM        |        92.8         |            91.9             |         91.4          | -->

## Contact

Thanks for your attention! If you have any suggestion or question, you can leave a message here or contact us directly:

- changdongliang@bupt.edu.cn
- mazhanyu@bupt.edu.cn

## Others 
Based on the fgvclib, we have developed an FGVC WeChat applet for fine-grained visual classification in practice, which can be accessed by searching "细粒度图像分类" in WeChat, and there is a demo: https://reurl.cc/rRZE7O.


## Citation
If you find this library useful in your research, please consider citing:
```
@misc{Chang2023,
  author = {Dongliang Chang, Ruoyi Du, Xinran Wang, Yuqi Yang, Yi-Zhe Song, Zhanyu Ma},
  title = {Fine-grained Visual Analysis Library},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PRIS-CV/Fine-grained-Visual-Analysis-Library}}
}
```



