CUDA_VISIBLE_DEVICES=0,1,2 python trainer_hf.py --exp_name audio_ce --using_model audio --batch_size 2 --accumulate_grad 8
CUDA_VISIBLE_DEVICES=0,1,2 python trainer_hf.py --exp_name both_ce  --using_model both --batch_size 2 --accumulate_grad 8
CUDA_VISIBLE_DEVICES=0,1,2 python trainer_hf.py --exp_name text_ce  --using_model text --batch_size 64 --accumulate_grad 1

CUDA_VISIBLE_DEVICES=0,1,2 python trainer_hf.py --exp_name audio_multiloss --using_model audio --batch_size 2 --accumulate_grad 8 --loss cs_and_ce
CUDA_VISIBLE_DEVICES=0,1,2 python trainer_hf.py --exp_name both_multiloss  --using_model both --batch_size 2 --accumulate_grad 8 --loss cs_and_ce
CUDA_VISIBLE_DEVICES=0,1,2 python trainer_hf.py --exp_name text_multiloss  --using_model text --batch_size 64 --accumulate_grad 1 --loss cs_and_ce

CUDA_VISIBLE_DEVICES=0,1,2 python trainer_hf_MMER.py --exp_name both_ce_MMI --batch_size 2 --accumulate_grad 8
CUDA_VISIBLE_DEVICES=0,1,2 python trainer_hf_MMER.py --exp_name both_multiloss_MMI --batch_size 2 --accumulate_grad 8 --loss cs_and_ce