如何執行程式？

(1)剪枝
命令列指令:
python main.py [dataset_location(ex. ~/dataset/imagenet)]
(train, val資料夾都在imagenet內)

結果:
1.得出瘦身後的模型
(以預設而言,將產出pruned_model1.pth~pruned_model11.pth及pruned_model_final.pth)
2.得出每個模型的結構
(trace_model1.pt~trace_model11.pt)(使用neutron解析即可得知模型結構)

(2)測量Accuracy
命令列指令:
python main.py -e [dataset_location(ex. ~/dataset/imagenet)]
(要有pruned_model1.pth, ... pruned_model11.pth, pruned_model_final.pth)

結果:
得出12個模型,每個模型的Accuracy


參數預設:
(可在main.py修改)
減少30%的latency(pruning_factor)
每次while iteration砍前次best_model的latency的1/10(init_red_fact)
batch_size = 256
learning_rate = 1e-4(0.0001)
"短期微調"用training data的1/3(256*1692 約等於 1280000張image的1/3)
"長期微調"用整個training data(256*5078 約等於 1280000張image)
print_freq=100 (print的頻率)

Written by Bobby. 2020/08/18