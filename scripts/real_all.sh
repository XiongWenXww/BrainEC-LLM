#!/bin/bash

# Base settings
path='../fMRI/DataSets_Feedbacks-selected/5.RestingState_MTL/'
data=Real
seed=42
epochs=300
lora_dropout=0.01
learning_rate=0.001
llm_layers=10
batch_size=2
d_model=32
d_ff=32
dropout=0.01
down_sampling_layers=3
alpha_sp=2
alpha_DAG=100
alpha_cons=10
comment='BrainECLLM-Sanchez'
llm_model='LLAMA3'
llm_path='Meta-Llama-3-8B-Instruct'

positions=("left" "right")

for pos in "${positions[@]}"
do
  echo "Running for dataset $data with path $path, pos $pos, alpha_DAG $alpha_DAG"

  log_path=./outlog/${data}/
  if [ ! -d "${log_path}" ];then
      mkdir $log_path
  fi

  nohup python -u run_main.py \
    --seed $seed \
    --path $path \
    --data $data \
    --index 1 \
    --pos $pos \
    --batch_size $batch_size \
    --lradj 'COS' \
    --learning_rate $learning_rate \
    --lower_bound 0.2 \
    --upper_bound 0.3 \
    --lora_dropout $lora_dropout \
    --d_model $d_model \
    --d_ff $d_ff \
    --dropout $dropout \
    --llm_model $llm_model \
    --llm_path $llm_path \
    --llm_layers $llm_layers \
    --down_sampling_method 'avg' \
    --down_sampling_layers $down_sampling_layers \
    --epochs $epochs \
    --model_comment $comment \
    --soft_threshold 0.5 \
    --alpha_sp $alpha_sp \
    --alpha_DAG $alpha_DAG\
    --alpha_cons $alpha_cons \
    > ${log_path}${pos}.log 2>&1 &

    wait $!
done