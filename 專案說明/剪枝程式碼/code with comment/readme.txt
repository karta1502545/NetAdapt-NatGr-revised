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
Dataset(train跟val)要放在起始頁的/dataset/imagenet裡面程式才跑得動

參數預設:
(可在main.py修改)
減少30%的latency(pruning_factor)
每次while iteration砍前次best_model的latency的1/10(init_red_fact)
batch_size = 256
learning_rate = 1e-4(0.0001)
"短期微調"用training data的1/3(256*1692 約等於 1280000張image的1/3)
"長期微調"用整個training data(256*5078 約等於 1280000張image)
print_freq=100 (print的頻率)

2.Evaluate a model only:
python main.py ~/dataset/imagenet -e 
--training_model=XXXX.pth (default: pruned_model_final.pth)

usage: python main.py ~/dataset/imagenet -e --training_model=XXXX.pth
