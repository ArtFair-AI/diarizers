batch_sizes=(32 64 128)
learning_rates=(1e-3 1e-5 1e-7)

for batch_size in "${batch_sizes[@]}"
do
    for learning_rate in "${learning_rates[@]}"
    do
        output_dir=fine_tuned_segmentation-3.0_${learning_rate}_${batch_size}
        echo "Training with batch size: $batch_size and learning rate: $learning_rate, output_dir: $output_dir"
        python3 train_segmentation.py \
            --dataset_name=ArtFair/diarizers_dataset_70-15-15 \
            --dataset_config_name=default \
            --model_name_or_path=pyannote/segmentation-3.0 \
            --output_dir=$output_dir \
            --do_train \
            --do_eval \
            --learning_rate=$learning_rate \
            --num_train_epochs=5 \
            --lr_scheduler_type=cosine \
            --per_device_train_batch_size=$batch_size \
            --per_device_eval_batch_size=32 \
            --evaluation_strategy=epoch \
            --save_strategy=epoch \
            --preprocessing_num_workers=2 \
            --dataloader_num_workers=2 \
            --logging_steps=100 \
            --load_best_model_at_end \
            --push_to_hub
    done
done