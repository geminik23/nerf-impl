# NeRF Implementations in PyTorch: Focus on Speed and Efficiency

The purpose of this repository is to gain a comprehensive understanding of NeRF and Volumetric Rendering, focusing on speed and efficiency of inference. The main objective is to adapt NeRF for real-time applications by experimenting with various accelerated models

## Current Implementation: FastNeRF

- Implemented the factorized model.
- Currently, working on integrating caching mechanism from [VDB: High-Resolution Sparse Volumes with Dynamic Topology](https://ken.museth.org/Publications_files/Museth_TOG13.pdf)

## Future Implementation Plans

- [KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs](https://arxiv.org/abs/2103.13744)
- [PlenOctrees for Real-time Rendering of Neural Radiance Fields](https://arxiv.org/abs/2103.14024)
- [Baking Neural Radiance Fields for Real-Time View Synthesis](https://arxiv.org/abs/2103.14645)


## Training

To train the different NeRF models, use the `train.py` script. The script will support multiple models.

```bash
python train.py --model nerf --config config.json
```


## Original NeRF Implementation in PyTorch

The entire implementation was initially done in Jupyter Notebooks, ensuring a step-by-step understanding and testing:

1. [1_camera.ipynb](1_camera.ipynb)
2. [2_load_dataset.ipynb](2_load_dataset.ipynb)
3. [3_volumetric_rendering.ipynb](3_volumetric_rendering.ipynb)
4. [4_voxel_reconstruction.ipynb](4_voxel_reconstruction.ipynb)
5. [5_train_nerf.ipynb](5_train_nerf.ipynb)


## Visualizations

Visual results achieved using the NeRF implementation on a chair dataset:

- The result of Voxel Reconstruction:

<img src="./img/out_voxel_chairs.gif" width="200">

- The result of NeRF Output:

<img src="./img/out_nerf_chairs.gif" width="200">

- Training Visualization:

<img src="./img/train_nerf_chair.gif" width="200">


## References

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934v2.pdf)
- [scratchapixel](https://www.scratchapixel.com/)