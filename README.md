# RaLL
RaLL: End-to-end Radar Localization on Lidar Map Using Differentiable Measurement Model

<img src="https://github.com/ZJUYH/RaLL/blob/master/img/robotcar.gif" width=500>

### data/maps (>80Mb)
point cloud maps and images with resolution of 0.25m/pixel
### data/gt_poses (>80Mb)
groud truth poses for evaluation
### data/odom
odometry data via ICP

### ekf_filter
differetiable ekf implementation
### loss
cross entropy loss (L1) and squared error loss (L2)
### network
feature extraction network and patch network
### test_py
pose tracking files on RobotCar and MulRan

### To train RaLL
Please use the `train_rall_L12.py` and `train_rall_L3.py`.
Please modify the data path in the python files.

If you use our data or code in an academic work, please cite the following paper:

	@article{yin2021rall,
	  title={RaLL: End-to-end Radar Localization on Lidar Map Using Differentiable Measurement Model},
	  author={Yin, Huan and Chen, Runjian and Wang, Yue and Xiong, Rong},
	  journal={IEEE Transactions on Intelligent Transportation Systems},
	  year={2021},
	  publisher={IEEE}
	}

If you have any questions, feel free to contact: [Huan Yin](https://yinhuan.site/) `zjuyinhuan@gmail.com`.

### TODO
- Upload the trained models.