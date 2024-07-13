accelerate launch train_text_to_image_lora.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" --dataset_name="/workspace/loraF" --resolution=512 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=15000 --learning_rate=1e-04 --max_grad_norm=1 --lr_scheduler="cosine" --lr_warmup_steps=0 --output_dir="/workspace/LORA_RESULTS" --checkpointing_steps=500  --validation_prompt="f4ke" --seed=1337
