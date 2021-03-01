#/home/gangwei.jgw/python/bin/python4 main.py --mode="DSNAS" --dataset="mushroom" --gen_max_child --early_fix_arch --arch_lr=0.01
python main.py --mode="DSNAS_v" --dataset="Adult" --gen_max_child --early_fix_arch --arch_lr=0.01
python main.py --mode="DSNAS_v" --dataset="Adult" --gen_max_child --early_fix_arch --arch_lr=0.01 --search_epoch=5
python main.py --mode="DSNAS_v" --dataset="Adult" --gen_max_child --early_fix_arch --arch_lr=0.01 --multi_operation
python main.py --mode="DSNAS_v" --dataset="Adult" --gen_max_child --early_fix_arch --arch_lr=0.01 --multi_operation --ofm
python main.py --mode="NASP_v" --dataset="Adult" --arch_lr=0.001 --multi_operation --seed=39

