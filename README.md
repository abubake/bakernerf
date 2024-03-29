# Howdy!
To get started, the requirements file contains everything that must be installed after cloning the repo. Simply run the following command to create it, we recommend installing within a conda environment you create:

<!-- Go to the yml file, change the name from /home/eherrin@ad.ufl.edu/code/baker_nerf/env to whatever your prefer. EX: bakernerf-env
conda env create -f env.yml
conda activate bakernerf-env -->

Then, create a conda environment and install the requirements within it.

```
conda env create myEnviroName
conda activate myEnviroName
pip install -r requirements.txt
```

# How to use this program:

The program main functionalities are in the jupyter .ipynb files:

* processing_training_data - Use this file to process training data you generated from blender or omniverse code, so that it can be used in Nerf-reconstruction
* Nerf-reconstruction - Program to train a NeRF given rays (can be used for unzipped data you already have)
* Testing - Use this to test PSNR in rendering novel views
* MeshExtraction - Use this to generate a 3D mesh from your trained model
* visualize - Use this to visualize pose and ray data for one or many poses

To run with an existing dataset, simply unzip it in the datasets folder. 

## Results:
![image](https://github.com/abubake/bakernerf/assets/32299736/1be68feb-dac2-4bb1-be88-19a7957f07dc)

![image](https://github.com/abubake/bakernerf/assets/32299736/3149ade5-f62e-4674-8e10-07e6d003f860)

