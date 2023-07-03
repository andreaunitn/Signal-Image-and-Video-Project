# Signal, Image and Video Project

Welcome to our project. Follow this read.me to understand how to execute our project on your machine.

## Environment setup

We suggest creating a virtual environment to install all the required Python packages in.
Use the following commands to install Virtuslenv and create a new environment.
```shell
python3 -m pip install virtualenv
python3 -m venv /path/to/virtual/environment/env_name 
```
To start the environment use
```shell
source env_name/bin/activate #on unix systems
source env_name/Scripts/activate #on Windows
```
To install the required packages use the reference file
```shell
python3 -m pip install -r requirementsCUDA.txt #for CUDA compatible systems
python3 -m pip install -r requirementsMAC.txt #for macOS silicon systems
```

## OpenReID installation

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

Remember to change triplet_loss default values like k
