# SyntheT-Less
Official repository for the SyntheT-Less dataset

![alt_text](animated_samples.gif)

# Data Generation code

*Code coming soon*

Make sure you have installed the following requirements:

- [Blender](https://www.blender.org/download/Blender2.80/blender-2.80-linux-glibc217-x86_64.tar.bz2/)
- OpenCV
- OpenEXR
- imageio

```
python3 call_blender_multi.py --blender_path BLENDER_PATH \
			      --dtd-rootdir DTD_PATH \
			      --models_path CAD_PATH \
			      --cpus NUM_CPUS --size DATASET_SIZE --cuda CUDA_DEVICE
```

# Citation
If you use our data generation code or already generated data, please our paper:

```
@article{pitteri2019threedv, 
 Title = {On Object Symmetries and 6D Pose Estimation from Images}, 
 Author = {G. Pitteri and M. Ramamonjisoa and S. Ilic and V. Lepetit}, 
 Journal = {International Conference on 3D Vision}, 
 Year = {2019}
 }
```
