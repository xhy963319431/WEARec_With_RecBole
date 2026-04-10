gpu_id=0

# Yelp
# python run_model.py --gpu_id=${gpu_id} --model=WEARec --dataset="yelp" --config_files="configs/yelp_WEARec.yaml"

# Beauty
# python run_model.py --gpu_id=${gpu_id} --model=WEARec --dataset="Amazon_Beauty" --config_files="configs/Amazon_Beauty_WEARec.yaml"

# # Sports
# python run_model.py --gpu_id=${gpu_id} --model=WEARec --dataset="Amazon_Sports_and_Outdoors" --config_files="configs/Amazon_Sports_and_Outdoors_WEARec.yaml"

# # Toys
python run_model.py --gpu_id=${gpu_id} --model=WEARec --dataset="Amazon_Toys_and_Games" --config_files="configs/Amazon_Toys_and_Games_WEARec.yaml"

