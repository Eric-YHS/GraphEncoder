

python scripts/sweep_layers.py --detach --gpus 0 --dec_range 5,5 --grid \
                               --encoder_name cls_graphormer --denoiser_name uni_o2_condition

python scripts/sweep_layers.py --detach --gpus 1 --dec_range 5,5 --grid \
                               --encoder_name cls_graphormer --denoiser_name uni_o2_cat

python scripts/sweep_layers.py --detach --gpus 2 --dec_range 5,5 --grid \
                               --encoder_name cls_gps --denoiser_name uni_o2_condition

python scripts/sweep_layers.py --detach --gpus 3 --dec_range 5,5 --grid \
                               --encoder_name cls_gps --denoiser_name uni_o2_cat

