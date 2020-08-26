PyTorch version:1.2.0

1.Train:
python main.py ~/dataset/imagenet 
-b=256                       # batch_size
--short_term_fine_tune=5078  # 1 epoch
--long_term_fine_tune=101560 # 20 epoch
--pruning_fact=0.3           # prune 30% of latency
--init_red_fact=20           # prune 1/20 * pruning_fact per while loop iteration
-j=10                        # # of workers
--gpu=1                      # gpu-id

main.py line113 可以蓋掉pretrained model, 也就是可以reload model做evaluate, train, 或者繼續prune

usage: python main.py ~/dataset/imagenet -b=256 --short_term_fine_tune=5078 --long_term_fine_tune=101560 --pruning_fact=0.3 --init_red_fact=20 -j=10--gpu=0

2.Evaluate a model only:
python main.py ~/dataset/imagenet -e 
--training_model=XXXX.pth (default: pruned_model_final.pth)

usage: python main.py ~/dataset/imagenet -e --training_model=XXXX.pth
