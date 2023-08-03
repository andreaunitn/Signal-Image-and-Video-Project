# Signal, Image and Video Project

Report: [SIV.pdf](https://github.com/andreaunitn/Signal-Image-and-Video-Project/files/12252926/SIV.pdf)
</br>

Follow this read.me to understand how to execute our Signal, Image and Video project on your machine.

## Environment setup

We suggest creating a virtual environment to install all the required Python packages in.
Use the following commands to install Virtuslenv and create a new environment.
```shell
python3 -m pip install virtualenv
python3 -m venv /path/to/virtual/environment/env_name 
```
To start the environment
```shell
source env_name/bin/activate #on unix systems
source env_name/Scripts/activate #on Windows
```
To exit the environment
```shell
deactivate
```

To install the required packages use the reference file
```shell
python3 -m pip install -r requirementsCUDA.txt #for CUDA compatible systems
python3 -m pip install -r requirementsMAC.txt #for macOS silicon systems
```

If you have a GPU (higly reccomended) install torch with hardware acceleration. It is mandatory beacuse otherwise the code does not work.

## Installation

To install the OpenReID library with our modifications and tricks, execute the following command in the main directory of the project. 
Please not that this operation is required every time a change to the project is done.

```shell
python3 setup.py install
```

## Execution

To train the model with every trick just use

```shell
cd tricks
python3 triplet_loss.py -t 6 --combine-trainval
```

Use the following options to modify the settings

```shell
-t _ #to select the tricks to use (up to that number)
-d _ #to select the dataset to use
-b _ #to select the batch the size
--epochs _ #to change the number of epochs
--num-instances _ #to select the number of image per identity
--cross-domain #to make the final test on the other dataset (dukemtmc / market1501)
--data-dir _ #to change the dataset directory
--logs-dir _ #to change the logs directory
--evaluate #to just execute the evaluation
--resume _ #to resume training from the given checkpoint


-j _ #to select the number of workers
--height _ #to change the height of the input images
--width _ #to change the width of the input images
--re-ranking #to use re-ranking
--combine-trainval #to use validation images during training
```
__Market1501__ and __DukeMTMC-reID__ can be downloaded [here](https://drive.google.com/drive/folders/1pTjMzG4aoc4MgSCrXbQocREQG_HDSMWq?usp=sharing).
