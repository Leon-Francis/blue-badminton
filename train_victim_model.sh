currTime=$(date +"%Y-%m-%d_%T")

nohup python ./victim_module/victim_train.py > victim_module/logs/victim_bert_${currTime}.log 2>&1 &