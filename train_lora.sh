


MODEL_DIR='/storage/panqihe/checkpoints/stable-diffusion/stable-diffusion-inpainting'
OUTPUT_DIR='exp/debug'
accelerate launch train_lora.py \
    --pretrained_inpaint_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --lora_rank=64 \
    --learning_rate=3e-7 --loss_type="huber" --adam_weight_decay=0.0 \
    --max_train_steps=50000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=16 \
    --validation_steps=10000 \
    --checkpointing_steps=10000 --checkpoints_total_limit=10 \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --seed=453645634 \
    --description='data thred 1/16-1/8' \