#### Prepare

1. Download the code and data from github, 'fm_nas' directory
2. Environment: torch,  coloredlogs,  sklearn


#### Running Example

1. AutoCO：

   python simulate.py --device cuda  --dim 8 --decay 0.001 --learning 0.005 --times 5  --s_batch 200 --iter 200000 --batch 10000  --alpha 0.0001 --epoch 150 --trick 2 --first_order 1 --model_ee fmts --model_nas nas --model_struct 3  --auc_record 1 

2. FM：

   python simulate.py --device cuda  --dim 8 --decay 0.001 --learning 0.005 --times 5  --s_batch 200 --iter 200000 --batch 10000  --alpha 0.0 --epoch 200 --oper multiply --first_order 1 --model_ee fmeg --model_nas fm --model_struct 0  --auc_record 1 

3. LinUCB：

   python4 simulate.py  --times 5  --s_batch 100 --iter 100000 --batch 10000  --alpha 0.01 --model linucb 

   

#### Parameters

+ model: policy/algorithmn
  + "fmeg"：    train the FM models with new data in, take e-greedy for exploaration
  + "fmts":     train the FM models with new data in, take thompson sampling for exploaration
  + "fmgreedy"：train the FM model with large number of impressions but with no update
  + "random":   randomly choose the creative
  + "greedy" :  choose the ground-thruth creative
  + "linucb" :  linucb
  + "ts":       linearTS
+ model_struct: {0:nas, 1:nas+fc, 2:nas+OAE, 3:nas+OAE+fc}
+ batch: intervals to update the model
+ s_batch: intervals to recommend the creative
+ learning: learning rate
+ iter: total impressions
+ alpha: parameters for exploration
+ epoch: training epochs
+ ts_trick: {0: set std for partial features, 1: set std for all features}
+ first_order: {0: without the first order, 1: with the first order}
+ calcu_dense: {0: no interactions between contiunous features, 1: interactions between continuous features}
+ trick: mode for training, {0: fix the architecture and train the embedding, 1:train both the arch and embs}
+ arch: if trick==0, input the arch for training
+ dense: a vector with size of 3, the first value means the length for continuous features, the second for the categorical features, the last for multi-hot
+ info_path: path for data
+ data_name: name for data
+ dim： embedding size
+ decay： the weight_decay for adam optimizer
+ data_size: in the case of model==fmgreedy, the number of impressions
+ updata: {0: search for the arch each batch, 1: periodically search the arch}




#### Results

​	save path  os.path.join(args.info_path, args.data_name,"res",params['model']+"-"+timestamp+'.txt')



#### Structure of codes

```python
Interaction/ 
├── dataset #the path of datasets
│   └── data_nas  #simulation data
│       ├── data_loader.py #generate data files
│       ├── creative_list.pkl #features for creatives
│       ├── feature_list.pkl #features for items
│       ├── feature_size.pkl #information about item features
│       ├── item_candidates.pkl #candidate creatives
│       ├── pro_list.pkl #the ratio of impressions for items
│       └── rand_0.15 # one example of ctr
│           ├── auc_res #auc
│           ├── item_ctr.pkl #ctr
│           ├── model 
│           ├── model_param.pkl
│           ├── random_log.pkl #random
│           └── res #auc results
├── fm_nas
│   ├── ctr.py #structure of data
│   ├── model.py #code of models
│   ├── policy.py #code of explorations
│   ├── simulate.py #main file for simulation
│   ├── train_arch.py #training code for nas
│   ├── train.py #training code for fm
│   ├── utils_gen.py #
│   └── utils.py 
```



#### Generation of data

1. download the online from dataset/data_nas/rawdata/download_log_online.url
2. preprocess the log. run data_loader.py

   1. dataset.pkl ： log
3. analyse the log and generate the data files for experiment. run data_loader.py

   1. creative_list.pkl :   the list of candidate creatives
   2. feature_list.pkl:     features for items
   3. item_candidates.pkl:  candidate creatives for each item
   4. pro_list.pkl:         the ratio of impressions for each item
4. train the FM models to fit the data 'dataset.pkl', obtain the ground truth parameters, run train.py and train_arch.py

   1. fm.pt
5. obtain the simulated CTR for each item according to the trained FM models, run utils_gen.py

   1. item_ctr.pkl
   2. model_param.pkl
6. randomly generate n impressions for training
   1. random_log.pkl
