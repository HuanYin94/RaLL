# RaLL
RaLL: End-to-end Radar Localization on Lidar Map Using Differentiable Measurement Model

Video:  [YouTube](https://youtu.be/a3wEv-eVlcg) | [Bilibili](https://www.bilibili.com/video/BV1my4y1b7Ns)

<img src="https://github.com/ZJUYH/RaLL/blob/master/img/robotcar.gif" width=500>

### Folders

* `data/maps (>80Mb)`

point cloud maps and images with resolution of 0.25m/pixel

* `data/gt_poses`

groud truth poses for evaluation

* `data/odom`

odometry data via ICP

* `ekf_filter`

differetiable ekf implementation

* `loader`

dataLoader for training and inference

* `loss`

cross entropy loss (L1) and squared error loss (L2)

* `models`

Pre-trained models of RaLL

* `network`

feature extraction network and patch network

* `test_py`

test pose tracking on RobotCar and MulRan

### To train RaLL
Please use the `train_rall_L12.py` and `train_rall_L3.py`.
Please modify the data path in the python files.

### Publication
If you use the data or code in an academic work, or inspired by our method, please consider citing the following:

	@article{yin2021rall,
	  title={RaLL: End-to-end Radar Localization on Lidar Map Using Differentiable Measurement Model},
	  author={Yin, Huan and Chen, Runjian and Wang, Yue and Xiong, Rong},
	  journal={IEEE Transactions on Intelligent Transportation Systems},
	  year={2021},
	  publisher={IEEE}
	}

We also propose a heterogeneous place recognition method with radar and lidar. Please refer to [Radar-to-Lidar](https://github.com/ZJUYH/radar-to-lidar-place-recognition) for details.
