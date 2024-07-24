DEBUG_MODE="-m debugpy --listen 127.0.0.1:5679 --wait-for-client"

model_name=Llama-2-7b-hf
task=webshop
worker_num=31

exp_name=$1

node_num=4  # number of GPUs
num_workers=4   # number of inference workers
sample_node_num=8
sample_num_workers=8

model_path=/home/azureuser/weimin/models/ # path to the original LLM
save_dir=/home/azureuser/weimin/agentpipeline/checkpoints_${task}/    # checkpoint save path
save_path=/home/azureuser/weimin/agentpipeline/experiments/${model_name}-${task}-sft-step-entire-monte-carlo-beta-0.1-lr3e-6/  # output save path
logs_path=${save_path}logs

if [ "$task" == "intercode_sql" ]; then
    docker stop docker-env-sql_ic_ctr
    docker rm docker-env-sql_ic_ctr
    bash setup_sql.sh
fi

if [ -d ${save_path} ]; then
    rm -r ${save_path}
fi
mkdir -p ${save_path}
mkdir -p ${logs_path}/

# Part 1: SFT stage
sft_data_path="data/${task}_sft.json"
batch_size=48
micro_batch_size=4
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

sft_model_name=${exp_name}${model_name}-${task}-sft-step-entire-monte-carlo-beta-0.1-lr3e-6

python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20002 fastchat/train/train.py \
    --model_name_or_path ${model_path}${model_name} \
    --data_path ${sft_data_path} \
    --bf16 True \
    --output_dir ${save_dir}${sft_model_name} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${accumulation_step} \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess False

# if failed, exit
if [ $? -ne 0 ]; then
    echo "SFT training failed"
    exit 1
fi

# Part 2: Evaluate SFT agent
fs_worker_port=21012
python -u -m fastchat.serve.vllm_worker --model-path ${save_dir}${sft_model_name} --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &

fs_worker_pid=$!
sleep 60

# evaluate on the test set
python -m eval_agent.main --agent_config fastchat --model_name ${sft_model_name} --exp_config ${task} --split test --override

# if failed, exit
if [ $? -ne 0 ]; then
    echo "base agent evaluation failed"
    kill -9 $fs_worker_pid
    exit 1
fi

# kill the model worker
kill -9 $fs_worker_pid

cur_model_name=${sft_model_name}
monte_carlo_explore_model_name=${cur_model_name}-monte-carlo-explore
for i in {1..6}; do
    # Part 3: Base agent explore stage
    # launch the fastchat model worker

    if [ "$task" == "intercode_sql" ]; then
        docker stop docker-env-sql_ic_ctr
        docker rm docker-env-sql_ic_ctr
        bash setup_sql.sh
        sleep 60
    fi

    explore_model_name=${cur_model_name}-explore

    for ((j=0;j<${sample_num_workers};j=j+1)); do
        if [ -d "${save_dir}${explore_model_name}-${j}" ]; then
            echo "Link to model exists"
        else
            ln -s ${save_dir}${cur_model_name} ${save_dir}${explore_model_name}-${j}
        fi
    done
    if [ -f "${logs_path}/worker_pid.txt" ]; then
        rm ${logs_path}/worker_pid.txt
    fi

    fs_worker_port=21012
    worker_idx=0
    for ((j=0;j<${sample_num_workers};j=j+1)); do
        echo "Launch the model worker on port ${fs_worker_port}"
        CUDA_VISIBLE_DEVICES=$((${worker_idx} % ${sample_node_num})) python -u -m fastchat.serve.vllm_worker \
            --model-path ${save_dir}${explore_model_name}-${j} \
            --port ${fs_worker_port} \
            --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/worker_pid.txt
        fs_worker_port=$(($fs_worker_port+1))
        worker_idx=$(($worker_idx+1))
        sleep 15
    done
    
    sleep 60

    # start explore on the same sft data
    echo "Base agent starts exploring"
    if [ -f "${logs_path}/eval_pid.txt" ]; then
        rm ${logs_path}/eval_pid.txt
    fi

    step_traj_save_path=${save_path}${explore_model_name}
    if [ -d ${step_traj_save_path} ]; then
        rm -r ${step_traj_save_path}
    fi
    mkdir -p ${step_traj_save_path}

    for (( j = 0; j <= $worker_num; j++ )); do
        python3 generate_response.py --exp_config ${task} --model_name ${explore_model_name}-$((j%sample_node_num)) --part_num $((worker_num+1)) --part_idx ${j} --save_path ${step_traj_save_path}  >> ${logs_path}/gen_response_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/eval_pid.txt
    done

    wait $(cat ${logs_path}/eval_pid.txt)
    rm ${logs_path}/eval_pid.txt
    echo "Base agent has finished exploring"

    # if failed, exit
    if [ $? -ne 0 ]; then
        echo "base agent exploration failed"
        kill -9 $(cat ${logs_path}/worker_pid.txt)
        rm ${logs_path}/worker_pid.txt
        exit 1
    fi

    # kill the model worker
    echo "Kill the model workers"
    kill -9 $(cat ${logs_path}/worker_pid.txt)
    rm ${logs_path}/worker_pid.txt

    # Part 4: Estimate step-level rewards

    for ((j=0;j<${sample_num_workers};j=j+1)); do
        if [ -d "${save_dir}${monte_carlo_explore_model_name}-${j}" ]; then
            echo "Link to model exists"
        else
            ln -s ${save_dir}${sft_model_name} ${save_dir}${monte_carlo_explore_model_name}-${j}
        fi
    done
    if [ -f "${logs_path}/worker_pid.txt" ]; then
        rm ${logs_path}/worker_pid.txt
    fi

    if [ "$task" == "intercode_sql" ]; then
        docker stop docker-env-sql_ic_ctr
        docker rm docker-env-sql_ic_ctr
        bash setup_sql.sh
        sleep 60
    fi

    fs_worker_port=21012
    worker_idx=0
    for ((j=0;j<${sample_num_workers};j=j+1)); do
        echo "Launch the model worker on port ${fs_worker_port}"
        CUDA_VISIBLE_DEVICES=$((${worker_idx} % ${sample_num_workers})) python -u -m fastchat.serve.vllm_worker \
            --model-path ${save_dir}${monte_carlo_explore_model_name}-${j} \
            --port ${fs_worker_port} \
            --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/worker_pid.txt
        fs_worker_port=$(($fs_worker_port+1))
        worker_idx=$(($worker_idx+1))
        sleep 15
    done
    sleep 60

    echo "Base agent starts monte carlo sampling"
    if [ -f "${logs_path}/eval_pid.txt" ]; then
        rm ${logs_path}/eval_pid.txt
    fi

    sample_num=5
    per_iteration_num=5
    sample_workers=16
    sample_iterations=$((sample_num/per_iteration_num))

    for ((j=0;j<${sample_iterations};j=j+1));do
        for ((k=0;k<${per_iteration_num};k=k+1)); do
            # Part 3: sample trajectories
            monte_carlo_sample_save_path=${save_path}monte_carlo_sample_iteration_${i}/sampled_traj_$((j*per_iteration_num+k))
            for ((l=0;l<$sample_workers; l++)); do
                output_path=${monte_carlo_sample_save_path}/
                if [ -d ${output_path} ]; then
                    rm -r ${output_path}
                fi
                mkdir -p ${output_path}
                python monte_carlo_sample_${task}.py --agent_config fastchat_explore --model_name ${monte_carlo_explore_model_name}-$((l%sample_num_workers)) --exp_config ${task} --part_num ${sample_workers} --part_idx ${l} --save_path ${output_path} --data_path ${step_traj_save_path} >> ${logs_path}/gen_response_worker-$((j*per_iteration_num+k))-${l}.log 2>&1 &
                echo $! >> ${logs_path}/eval_pid.txt
            done
        done
        wait $(cat ${logs_path}/eval_pid.txt)
        rm ${logs_path}/eval_pid.txt
        echo "Base agent has finished exploring ${j} iteration"
    done



    # kill the model worker
    echo "Kill the model workers"
    kill -9 $(cat ${logs_path}/worker_pid.txt)
    rm ${logs_path}/worker_pid.txt


    # Part 5: Build contrastive action pairs
    echo "Build preference data"
    pm_data_path=${save_path}data_pm/${task}_${exp_name}_pm_${i}.json
    
    if [ ! -d ${save_path}data_pm ]; then
        mkdir -p ${save_path}data_pm
    fi

    python construct_preference_monte_carlo_${task}.py --task $task --output_path ${pm_data_path} --traj_path ${step_traj_save_path} --sample_path ${save_path}monte_carlo_sample_iteration_${i} --global_traj --local_traj --traj_threshold 0.01 --step_threshold 0.01

    # Part 6: Conduct mixture trajectory optimization to learn from incorrect actions
    
    batch_size=48
    micro_batch_size=2
    node_num=8
    accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))
    beta=0.1
    lr=3e-6

    if [ "$task" == "intercode_sql" ]; then
        docker stop docker-env-sql_ic_ctr
        docker rm docker-env-sql_ic_ctr
        bash setup_sql.sh
        sleep 60
    fi
    
    dpo_model_name=${sft_model_name}-dpo-iter-${i}

    python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20002 fastchat/train/train_dpo.py \
        --model_name_or_path ${save_dir}${cur_model_name} \
        --ref_model_name_or_path ${save_dir}${sft_model_name} \
        --data_path ${pm_data_path} \
        --bf16 True \
        --output_dir ${save_dir}${dpo_model_name} \
        --num_train_epochs 3 \
        --per_device_train_batch_size ${micro_batch_size} \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps ${accumulation_step} \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --save_total_limit 5 \
        --beta ${beta} \
        --learning_rate ${lr} \
        --weight_decay 0. \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 5 \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True \
        --model_max_length 4096 \
        --max_prompt_length 512 \
        --max_target_length 3072 \
        --gradient_checkpointing True \
        --lazy_preprocess False

    # Part 6: Evaluate the agent
    fs_worker_port=21012
    python -u -m fastchat.serve.vllm_worker --model-path ${save_dir}${dpo_model_name} --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &

    fs_worker_pid=$!
    sleep 60

    # evaluate on the test set
    python -m eval_agent.main --agent_config fastchat --model_name ${dpo_model_name} --exp_config ${task} --split test

    # if failed, exit
    if [ $? -ne 0 ]; then
        echo "base agent evaluation failed"
        kill -9 $fs_worker_pid
        exit 1
    fi

    # kill the model worker
    kill -9 $fs_worker_pid

    cur_model_name=${dpo_model_name}
done
