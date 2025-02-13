# MultiAir
This is the official implementation of __MultiAir__ (Multi-modal model based on time series and satellite images for Air pollution nowcasting in extreme case), which is described in the following paper:
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

## How to run MultiAir
Carefully read readme.txt in the data folder, and complete the required process.
After that, with all the dependencies are installed, you can run MultiAir by running 
```
bash run.sh
```
which runs 
```
python src/sat_model_img_extract.py --model_name SimVP_pretrained --gpus 0,1 --input_dim 13 --output_dim 6 

python src/training.py --use_focal_loss --use_focal_L1 --model_name multiair_1hr_sample --max_time 1 --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name multiair_2hr_sample --max_time 2 --use_pretrained --pretrained_model multiair_1hr_sample --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name multiair_3hr_sample --max_time 3 --use_pretrained --pretrained_model multiair_2hr_sample --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name multiair_4hr_sample --max_time 4 --use_pretrained --pretrained_model multiair_3hr_sample --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name multiair_5hr_sample --max_time 5 --use_pretrained --pretrained_model multiair_4hr_sample --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name multiair_6hr_sample --max_time 6 --use_pretrained --pretrained_model multiair_5hr_sample --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0

python src/evaluation.py --model_name multiair_6hr_sample --gpus 0 --prev_len 7 --input_dim 7 --output_dim 6 --batch_size 24
```

_sat_model_img_extract.py_ takes raw satellite images, and generates observation & prediction numpy files for MultiAir, with following key parameters:
* [model_name]: name of the checkpoint for pretrained SimVP model
* [gpus]: specify the GPUs to use
* [input_dim]: length of input image sequence
* [output_dim]: maximum lead time
To modify the time range of satellite image to use (default 2021.01.01.00 - 2021.03.31.00) please modify
```
start_t = datetime(2021, 1, 1, 0, 0)
end_t = datetime(2021, 3, 31, 23, 0)
```

_training.py_ trains MultiAir with curriculum learning, with following key parameters:
* [model_name]: name for current training model
* [max_time]: current maximum lead time (for curriculum learning)
* [pretrained_model]: name of the checkpoint to load
* [gpus]: specify the GPUs to use
* [prev_len]: previous length for normalization method
* [input_dim]: length of input time-series sequence
* [output_dim]: maximum lead time

_evaluation.py_ evaluates MultiAir, with following key parameters:
* [model_name]: name of the checkpoint to evaluate
* [gpus]: specify the GPUs to use
* [prev_len]: previous length for normalization method
* [input_dim]: length of input time-series sequence
* [output_dim]: maximum lead time
