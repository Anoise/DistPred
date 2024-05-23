export CUDA_VISIBLE_DEVICES=6

if [ $# -eq 1 ]; then
  model_name=$1
else
  model_name=Minusformer
fi
echo $model_name

root_path=/home/user/daojun/Data/TS/illness
seq_len=36
dataset=ILI

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path national_illness.csv \
  --model_id ili_36_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'24.log

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path national_illness.csv \
  --model_id ili_36_36 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'36.log

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path national_illness.csv \
  --model_id ili_36_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'48.log

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path national_illness.csv \
  --model_id ili_36_60 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 >logs/$model_name'_'$dataset'_'$seq_len'_'60.log