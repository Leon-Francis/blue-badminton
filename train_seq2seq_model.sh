currTime=$(date +"%Y-%m-%d_%T")

nohup python ./train_seq2seq.py > logs/Seq2Seq_${currTime}.log 2>&1 &