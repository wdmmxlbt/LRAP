
#!/usr/bin bash

method="lrap"
prefix="outs"
suffix="lrap_pretrain"

for dataset in "clinc"
do
    for known_cls_ratio in 0.25 0.5
    do
        for seed in 0 
        do 
            python run.py \
            --dataset $dataset \
            --method ${method} \
            --known_cls_ratio $known_cls_ratio \
            --seed $seed \
            --config_file_name "./methods/${method}/configs/config_clinc.yaml" \
            --save_results \
            --log_dir "./${prefix}/${method}/logs/${dataset}_${suffix}" \
            --output_dir "./${prefix}/${method}" \
            --dataset_dir "datasets_${known_cls_ratio}/data_${known_cls_ratio}_${dataset}_${seed}.pkl" \
            --model_file_name "model_${known_cls_ratio}_${dataset}_${seed}_${suffix}.pt" \
            --result_dir "./${prefix}/${method}/results_${dataset}_${suffix}" \
            --results_file_name "results_${known_cls_ratio}_${dataset}_${seed}_${suffix}.csv"  \
            --cl_loss_weight 1.0 \
            --semi_cl_loss_weight 1.0
        done
    done
done


