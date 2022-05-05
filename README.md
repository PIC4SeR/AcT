[![arXiv](http://img.shields.io/badge/arXiv-2001.09136-B31B1B.svg)](https://arxiv.org/abs/2107.00606)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<h1 align="center">  Action Transformer <br> A Self-Attention Model for Short-Time Human Action Recognition
</h1>

<p align="center">
  <img src="https://ars.els-cdn.com/content/image/1-s2.0-S0031320321006634-gr1_lrg.jpg" alt="AcT Summary" width="450"/>
</p>

This repository contains the official TensorFlow implementation of the paper "Action Transformer: A Self-Attention Model for Short-Time Pose-Based Human Action Recognition".

Action Transformer (AcT), a simple, fully self-attentional architecture that consistently outperforms more elaborated networks that mix convolutional, recurrent and attentive layers. In order to limit computational and energy requests, building on previous human action recognition research, the proposed approach exploits 2D pose representations over small temporal windows, providing a low latency solution for accurate and effective real-time performance. 

To do so, we open-source [MPOSE2021](https://github.com/PIC4SeRCentre/MPOSE2021), a new large-scale dataset, as an attempt to build a formal training and evaluation benchmark for real-time, short-time HAR. MPOSE2021 is developed as an evolution of the MPOSE Dataset [1-3]. It is made by human pose data detected by 
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) [4] and [Posenet](https://github.com/google-coral/project-posenet/tree/master/models) [5]
on popular datasets for HAR.

<p align="center">
  <img src="https://ars.els-cdn.com/content/image/1-s2.0-S0031320321006634-gr6_lrg.jpg" alt="AcT Results" width="750"/>
</p>

This repository allows to easily run a benchmark of AcT models using MPOSE2021, as well as executing a random hyperparameter search. 

## Usage
First, clone the repository and install the required pip packages (virtual environment recommended!).

```
pip install -r requirements.txt
```

To run a random search:
```
python main.py -s
```

To run a benchmark:
```
python main.py -b
```

That's it!

This code uses the [mpose](https://pypi.org/project/mpose/) pip package, a friendly tool to download and process MPOSE2021 pose data.

# Citations
AcT is intended for scientific research purposes.
If you want to use this repository for your research, please cite our work ([Action Transformer: A Self-Attention Model for Short-Time Pose-Based Human Action Recognition](https://arxiv.org/abs/2107.00606)) as well as [1-5].

```
@article{mazzia2021action,
  title={Action Transformer: A Self-Attention Model for Short-Time Pose-Based Human Action Recognition},
  author={Mazzia, Vittorio and Angarano, Simone and Salvetti, Francesco and Angelini, Federico and Chiaberge, Marcello},
  journal={Pattern Recognition},
  pages={108487},
  year={2021},
  publisher={Elsevier}
}
```

# References
[1] Angelini, F., Fu, Z., Long, Y., Shao, L., & Naqvi, S. M. (2019). 2D Pose-Based Real-Time Human Action Recognition With Occlusion-Handling. IEEE Transactions on Multimedia, 22(6), 1433-1446.

[2] Angelini, F., Yan, J., & Naqvi, S. M. (2019, May). Privacy-preserving Online Human Behaviour Anomaly Detection Based on Body Movements and Objects Positions. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 8444-8448). IEEE.

[3] Angelini, F., & Naqvi, S. M. (2019, July). Joint RGB-Pose Based Human Action Recognition for Anomaly Detection Applications. In 2019 22th International Conference on Information Fusion (FUSION) (pp. 1-7). IEEE.

[4] Cao, Z., Hidalgo, G., Simon, T., Wei, S. E., & Sheikh, Y. (2019). OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields. IEEE transactions on pattern analysis and machine intelligence, 43(1), 172-186.

[5] Papandreou, G., Zhu, T., Chen, L. C., Gidaris, S., Tompson, J., & Murphy, K. (2018). Personlab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 269-286).

[6] Mazzia, V., Angarano, S., Salvetti, F., Angelini, F., & Chiaberge, M. (2021). Action Transformer: A Self-Attention Model for Short-Time Pose-Based Human Action Recognition. Pattern Recognition, 108487.



<p align="center">
  <img src="https://raw.githubusercontent.com/PIC4SeR/MPOSE2021_Dataset/master/docs/giphy.gif" alt="animated" />
</p>
