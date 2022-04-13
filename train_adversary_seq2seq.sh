currTime=$(date +"%Y-%m-%d_%T")

nohup python ./adversary_train.py > attack_logs/attack_${currTime}.log 2>&1 &