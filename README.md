# MTD
This is the code for our paper: Graph-Regularized Multi-Task Learning of Multi-Level Transition Dynamics for Session-based Recommendation. We have implemented our methods in Tensorflow

Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder `datasets/`:
- YOOCHOOSE: <http://2015.recsyschallenge.com/challenge.html>

- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup>

## Usage
You need to run the file  `datasets/preprocess_v1.py` first to preprocess the data.

For example: `cd datasets; python preprocess_v1.py --dataset=diginetica`

Then you can run the file `main.py` to train the model

```bash
usage: main.py [-h] [--dataset DATASET] [--method METHOD] [--validation]
               [--epoch EPOCH] [--batch_size BATCH_SIZE]
               [--hidden_size HIDDEN_SIZE] [--emb_size EMB_SIZE] [--l2 L2]
               [--lr LR] [--nonhybrid] [--lr_dc LR_DC]
               [--lr_dc_step LR_DC_STEP] [--dropout DROPOUT] [--rec REC]
               [--kg KG] [--rand_seed RAND_SEED] [--log_file LOG_FILE]
               [--trunc TRUNC] [--num_head NUM_HEAD] [--num_block NUM_BLOCK]
               [--num_gcn NUM_GCN] [--use_bias] [--save_tk]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name:
                        diginetica/yoochoose1_4/yoochoose1_64/sample
  --method METHOD       sa
  --validation          validation
  --epoch EPOCH         number of epochs to train for
  --batch_size BATCH_SIZE
                        input batch size
  --hidden_size HIDDEN_SIZE
                        hidden state size
  --emb_size EMB_SIZE   hidden state size
  --l2 L2               l2 penalty
  --lr LR               learning rate
  --nonhybrid           global preference
  --lr_dc LR_DC         learning rate decay rate
  --lr_dc_step LR_DC_STEP
                        the number of steps after which the learning rate
                        decay
  --dropout DROPOUT     dropout rate
  --rec REC             the frequence of recommendation loss
  --kg KG               the max number of dgi optimizer
  --rand_seed RAND_SEED
  --log_file LOG_FILE
  --trunc TRUNC         stop the dgi
  --num_head NUM_HEAD   number of self attention multi-head
  --num_block NUM_BLOCK
                        number of self attention block
  --num_gcn NUM_GCN     number of gcn layer
  --use_bias            use bias
  --save_tk             save recommendation result
```


## Requirements
python 3.6

tensorflow 1.12.0

networkx 1.11

## Implementation Reference
https://github.com/eliorc/node2vec

https://github.com/CRIPAC-DIG/SR-GNN
