## Setup and install

The  given softwares in the Softwares_req.txt needs to be installed.

To install the conda requirements for the tensorflow  gpu run
conda install --yes --file requirements.txt

To install the tensornets requirements for the tensorflow run
pip install -r pip_requirements.txt

## Basis Implementation

# 1. Downloading data
# 2. Basic Preprocessing/preparing our data
# 3. Set up loss functions
# 4. Create model
# 5. Optimize for loss function
#

## Main files used :- Main.py,  gatyes.py, Neural_patches.py , Semantic.py

Main.py is the main file and is used to navigate the project.

1) Gateys:-
To run python3 gatyes.py  --content samples/ck.jpg --style  samples/Renoir.jpg --iterations 1 --device cpu --folder samples
To run python3 Neural_Patches.py  --content samples/ck.jpg --style  samples/Renoir.jpg --iterations 1 --device cpu --folder samples
To run python3 Sematic.py --style style/the_scream.jpg --content samples/ck.jpg --folder=samples --device=cpu --iterations=100


The Preprocessing folder contains baisc files for the utiilities:-
utils.py

The images to test are stored in samples folder (images with semantic maps ), content_folder(  content images ), style_folder(  style_images ) :-
The semantic images need to created using the installed Pixel ....

The result images are stroed in model_name_output/input folder/image_name_res.png,
and the intermediate images are stored in model_name_output/input folder/image_name_inter.png

Example:-
The


## Neural Doodles
The Neural doodles can be formed using the

