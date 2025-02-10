# AIRTIME
This is the official implementation of __AIRTIME__ (AIR pollution nowcasting based on Time series and satellite Image Multimodal model for Extreme air pollution), which is described in the following paper:
* __Air Pollution Nowcasting via Multimodal Learning with Temporal and Visual Data__
<br> Sojeong Kim\*, Jaewan Chun\*, Jongha Lee\*, Sung-Chul Hong, Jae-Bum Lee, Seung-Hee Eun, Yonghee Lee, and Kijung Shin

The paper is accessible in the following link:
* TBD
## __Overview__
TODO

## Dependencies
* torch
* numpy
* tqdm
* argparse
* pyproj

## How to run AIRTIME
With all the dependencies installed, you can run AIRTIME by running 
```
bash run.sh
```
which runs 
```
python src/sat_model_img_extract.py --model_name SimVP_pretrained --gpus 0,1 --input_dim 13 --output_dim 6 

python src/training.py --use_focal_loss --use_focal_L1 --model_name airtime_1hr --max_time 1 --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name airtime_2hr --max_time 2 --use_pretrained --pretrained_model airtime_1hr --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name airtime_3hr --max_time 3 --use_pretrained --pretrained_model airtime_2hr --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name airtime_4hr --max_time 4 --use_pretrained --pretrained_model airtime_3hr --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name airtime_5hr --max_time 5 --use_pretrained --pretrained_model airtime_4hr --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name airtime_6hr --max_time 6 --use_pretrained --pretrained_model airtime_5hr --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0

python src/evaluation.py --model_name airtime_6hr --gpus 0 --prev_len 7 --input_dim 7 --output_dim 6 --batch_size 24
```

_sat_model_img_extract.py_ takes raw satellite images, and generates observation & prediction numpy files for AIRTIME, with following key parameters:
* [model_name]: name of the checkpoint for pretrained SimVP model
* [gpus]: specify the GPUs to use
* [input_dim]: length of input image sequence
* [output_dim]: maximum lead time

_training.py_ trains AIRTIME with curriculum learning, with following key parameters:
* [model_name]: name for current training model
* [max_time]: current maximum lead time (for curriculum learning)
* [pretrained_model]: name of the checkpoint to load
* [gpus]: specify the GPUs to use
* [prev_len]: previous length for normalization method
* [input_dim]: length of input time-series sequence
* [output_dim]: maximum lead time

_evaluation.py_ evaluates AIRTIME, with following key parameters:
* [model_name]: name of the checkpoint to evaluate
* [gpus]: specify the GPUs to use
* [prev_len]: previous length for normalization method
* [input_dim]: length of input image sequence
* [output_dim]: maximum lead time
