python -W ignore predict.py \
--ligand ZN \
--checkpoints_dir Dataset/result/ZN \
--device cuda:0 \
--radius 6.0 \
--n_layers 4 \
--input_feature_type esm chemi pka \
--batch_size 1 \
--lr 0.001 \
--loss_type site
