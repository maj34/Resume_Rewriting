CUDA_VISIBLE_DEVICES=0 python3.8 fine_tuning_with_qlora.py \
                                category=humanities \
                                run_name=humanities_batch_8_lr_0.00005_r_8_dropout_0.1 \
                                dataset=data/filtered_data_humanities.csv \
                                config_file=config/EEVE-Korean-10.8B-v1.0.yaml \
                                checkpoint_path=humanities_batch_8_lr_0.00005_r_8_dropout_0.1 \