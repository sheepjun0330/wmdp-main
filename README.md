Forget bio-wmdp

CUDA_VISIBLE_DEVICES=4,5 python -m rmu.unlearn --model_name_or_path HuggingFaceH4/zephyr-7b-beta --max_num_batches 150 --batch_size 4 --retain_corpora wikitext --forget_corpora bio-forget-corpus --steering_coeffs 6.5,6.5 --alpha 1200,1200 --lr 5e-6 --seed 42 --output_dir models/zephyr_rmu_alm_sam_sam_1e-5 --verbose --dual_mode alm_sam_sam_joint2 --tau 0.01 --lagran_lambda_init 0.0 --lagran_lambda_lr 1e-3 --forget_rho 1e-5 --retain_rho 1e-5 --use_wandb --wandb_project rmu-unlearn --wandb_run_name zephyr_rmu_alm_sam_sam1e-5_bio_tau0.01
