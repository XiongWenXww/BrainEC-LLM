path='../fMRI/DataSets_Feedbacks-selected/1.Simple_Networks/'
data=Sanchez
index=2
seed=42
epochs=95
lora_dropout=0.01
learning_rate=0.001
llm_layers=18
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
llm_path='/root/autodl-tmp/PretrainedModel/Meta-Llama-3-8B-Instruct'

log_path=./outlog/${data}/
if [ ! -d "${log_path}" ];then
    mkdir -p $log_path
fi

nohup python -u run_main.py \
  --seed $seed\
  --path $path\
  --data $data \
  --index $index\
  --batch_size $batch_size \
  --lradj 'COS'\
  --learning_rate $learning_rate \
  --lower_bound 0.2\
  --upper_bound 0.3\
  --lora_dropout $lora_dropout\
  --d_model $d_model \
  --d_ff $d_ff \
  --dropout $dropout\
  --llm_model $llm_model \
  --llm_path $llm_path \
  --llm_layers $llm_layers \
  --down_sampling_method 'avg' \
  --down_sampling_layers $down_sampling_layers\
  --epochs $epochs \
  --model_comment $comment \
  --soft_threshold 0.5 \
  --alpha_sp $alpha_sp \
  --alpha_DAG $alpha_DAG\
  --alpha_cons $alpha_cons \
  > ${log_path}sim${index}.log 2>&1 &
tail -f ${log_path}sim${index}.log