### Run the following command for intial check
```
python .\run_longExp.py --is_training 1 --root_path ./dataset --data_path 5m_intraday_data.csv --model_id ETTm2_$seq_len'_'96 --data trading --features M --seq_len 96 --pred_len 96 --enc_in 7 --des 'Exp' --loss MAE --tree_loss Huber --num_leaves 7 --tree_lr 0.01 --tree_iter 175 --normalize --tree_lb 96 --lb_data N --itr 1 --batch_size 32 --learning_rate 0.001 --model LTBoost --target Close  
```
