python src/sat_model_img_extract.py --model_name SimVP_pretrained --gpus 0,1 --input_dim 13 --output_dim 6 

python src/training.py --use_focal_loss --use_focal_L1 --model_name multiair_1hr_sample --max_time 1 --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name multiair_2hr_sample --max_time 2 --use_pretrained --pretrained_model multiair_1hr_sample --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name multiair_3hr_sample --max_time 3 --use_pretrained --pretrained_model multiair_2hr_sample --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name multiair_4hr_sample --max_time 4 --use_pretrained --pretrained_model multiair_3hr_sample --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name multiair_5hr_sample --max_time 5 --use_pretrained --pretrained_model multiair_4hr_sample --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0
python src/training.py --use_focal_loss --use_focal_L1 --model_name multiair_6hr_sample --max_time 6 --use_pretrained --pretrained_model multiair_5hr_sample --gpus 0,1 --batch_size 120 --prev_len 7 --input_dim 7 --output_dim 6 --seed 0

python src/evaluation.py --model_name multiair_6hr_sample --gpus 0 --prev_len 7 --input_dim 7 --output_dim 6 --batch_size 24
