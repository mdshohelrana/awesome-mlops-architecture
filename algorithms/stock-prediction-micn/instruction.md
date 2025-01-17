### Run the Following command
```
python .\run.py --model micn --mode regre --data Trading --features S --freq h --conv_kernel 12 16 --d_layers 1 --d_model 512 --seq_len 96 --data_path 5m_intraday_data.csv --target Close --enc_in 1 --dec_in 1 --c_out 1 --label_len 96 --pred_len 96
```
