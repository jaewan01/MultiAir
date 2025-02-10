# python src/sat_model_img_extract.py --model_name SimVP_pretrained --gpus 0,1 --input_dim 13 --output_dim 6 

# python src/training.py --use_focal_loss --use_focal_L1 --model_name airtime_1hr --max_time 1 --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
# python src/training.py --use_focal_loss --use_focal_L1 --model_name airtime_2hr --max_time 2 --use_pretrained --pretrained_model airtime_1hr --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
# python src/training.py --use_focal_loss --use_focal_L1 --model_name airtime_3hr --max_time 3 --use_pretrained --pretrained_model airtime_2hr --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
# python src/training.py --use_focal_loss --use_focal_L1 --model_name airtime_4hr --max_time 4 --use_pretrained --pretrained_model airtime_3hr --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
# python src/training.py --use_focal_loss --use_focal_L1 --model_name airtime_5hr --max_time 5 --use_pretrained --pretrained_model airtime_4hr --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
# python src/training.py --use_focal_loss --use_focal_L1 --model_name airtime_6hr --max_time 6 --use_pretrained --pretrained_model airtime_5hr --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/evaluation.py --model_name airtime_6hr --gpus 0,1 --prev_len 7 --input_dim 7 --output_dim 6 --batch_size 24