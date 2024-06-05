# True_gradient
This repository is the official implementation of **Theoretical understanding of gradients of spike functions as boolean functions**.

## Environment
Create virtual environment:
```setup
conda create -n name python=3.8.12
conda activate name
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```
To install requirements:
```setup
pip install -r requirements.txt
```
## Training
- Download 'dvs128-gestures.hdf5' file from [link](https://drive.google.com/file/d/12T0IhrZxhNakf3gjBlwJm698f88jNE2E/view?usp=drive_link)
- Save 'dvs128-gestures.hdf5' file to the 'true_gradient/dataset/DVS128-Gesture' directory

```train
python main_training.py
```
