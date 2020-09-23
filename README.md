This repository contains source code for our EMNLP 2020 paper "Reusing a Pretrained Language Model on Languages with Limited Corpora for Unsupervised NMT"  [(Paper link)](https://arxiv.org/abs/2009.07610)

# Introduction 

This paper presents a method to fine-tune of a pretrained monolingual LM (on a high-resource language) to a low-resource language,
 in order to serve as initialization of an Unsupervised NMT (UNMT) encoder-decoder model. 
 To this end, we propose a vocabulary extension method to allow fine-tuning. 
 
Our method, entitled **RE-LM**, provides very competitive UNMT results in low-resource - 
high-resource language pairs, outperforming XLM  in English-Macedonian (En-Mk) and English-Albanian (En-Sq), 
yielding more than +8.3 BLEU points for all four translation directions.

#### Reference

```
@misc{alex2020reusing,
    title={Reusing a Pretrained Language Model on Languages with Limited Corpora for Unsupervised NMT},
    author={Alexandra Chronopoulou and Dario Stojanovski and Alexander Fraser},
    year={2020},
    eprint={2009.07610},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

This code is built on using the [XLM](https://github.com/facebookresearch/XLM) baseline, which is publicly available. 

# Prerequisites 

#### Dependencies

- Python 3.6.9
- [NumPy](http://www.numpy.org/) (tested on version 1.15.4)
- [PyTorch](http://pytorch.org/) (tested on version 1.2.0)
- [Apex](https://github.com/NVIDIA/apex#quick-start) (for fp16 training)

#### Install Requirements 
**Create Environment (Optional):**  Ideally, you should create a conda environment for the project.

```
conda create -n relm python=3.6.9
conda activate relm
```

Install PyTorch ```1.2.0``` with the desired cuda version to use the GPU:

``` conda install pytorch==1.2.0 torchvision -c pytorch```

Clone the project:

```
git clone https://github.com/alexandra-chron/relm_unmt.git

cd relm_unmt
```


Then install the rest of the requirements:

```
pip install -r ./requirements.txt
```


To [train with multiple GPUs](https://github.com/facebookresearch/XLM#how-can-i-run-experiments-on-multiple-gpus) use:
```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
```

#### Download data 
We sample 68M English sentences from [Newscrawl](http://data.statmt.org/news-crawl/en/).

We use Macedonian and Albanian Common Crawl deduplicated monolingual data from the [OSCAR corpus](https://oscar-corpus.com/).

Our validation and test data is created by sampling from the  [SETIMES](http://opus.nlpl.eu/SETIMES.php) parallel En-Mk, En-Sq corpora. 
To allow reproducing our results, we provide the validation and test data in `./data/mk-en` and `./data/sq-en` directories.
## Preprocessing for pretraining

Before pretraining an HMR (high-monolingual-resource) monolingual MLM, make sure you
 have downloaded the HMR data and placed it in `./data/$HMR/` directory. 
 
 The data should be in the form:  `{train_raw, valid_raw, test_raw}.$HMR`. 

After that, run the following (example for En):
```
./get_data_mlm_pretraining.sh --src en
```


## RE-LM 

### 1. Train a monolingual LM 
Train your monolingual masked LM (BERT without the next-sentence prediction task) on the monolingual data:

```

python train.py                            \
--exp_name mono_mlm_en_68m                 \
--dump_path ./dumped                       \
--data_path ./data/en/                     \
--lgs 'en'                                 \
--mlm_steps 'en'                           \
--emb_dim 1024                             \ 
--n_layers 6                               \
--n_heads 8                                \
--dropout '0.1'                            \
--attention_dropout '0.1'                  \
--gelu_activation true                     \
--batch_size 32                            \
--bptt 256                                 \
--optimizer 'adam,lr=0.0001'               \
--epoch_size 200000                        \
--validation_metrics valid_en_mlm_ppl      \
--stopping_criterion 'valid_en_mlm_ppl,10' \

## There are other parameters that are not specified here (see train.py).
```

## Preprocessing for fine-tuning (and UNMT)

Before fine-tuning the pretrained MLM and running UNMT, make sure you
 have downloaded the LMR data and placed it in `./data/$LMR-$HMR/` directory. 
 
 The data should be in the form:  `{train_raw, valid_raw, test_raw}.$LMR`. 
 
 Then, run the following (example for En, Mk):
```
./get_data_and_preprocess.sh --src en --tgt mk
```

In Step 2, the embedding layer (and the output layer) of the MLM model will be increased by the amount of 
new items added to the existing vocabulary. 

In the directory `./data/$LMR-$HMR/`, a file named `vocab.$LMR-$HMR-ext-by-$NUMBER` has been created. 
This number indicates by how many items we need to extend the initial vocabulary, and consequently 
the embedding and linear layer, to account for the LMR language. 

You will need to give this value to the `--increase_vocab_by` argument so that you successfully run fine-tuning (step 2).  


### 2. Fine-tune it on both the LMR and HMR languages

```
python train.py                            \ 
--exp_name finetune_en_mlm_mk              \ 
--dump_path ./dumped/                      \ 
--reload_model 'mono_mlm_en_68m.pth'       \ 
--data_path ./data/mk-en/                  \
--lgs 'en-mk'                              \ 
--mlm_steps 'mk,en'                        \
--emb_dim 1024                             \
--n_layers 6                               \ 
--n_heads 8                                \
--dropout 0.1                              \
--attention_dropout 0.1                    \ 
--gelu_activation true                     \
--batch_size 32                            \
--bptt 256                                 \
--optimizer adam,lr=0.0001                 \
--epoch_size 50000                         \
--validation_metrics valid_mk_mlm_ppl      \
--stopping_criterion valid_mk_mlm_ppl,10   \
--increase_vocab_for_lang en               \
--increase_vocab_from_lang mk              \
--increase_vocab_by NUMBER (see ./data/mk-en/vocab.mk-en-ext-by-$NUMBER)
```

###3. Train a UNMT model (encoder and decoder initialized with RE-LM)

```
python train.py                            \
--exp_name unsupMT_ft_mk                   \
--dump_path ./dumped/                      \
--reload_model 'finetune_en_mlm_mk.pth,finetune_en_mlm_mk.pth' \
--data_path './data/mk-en'                 \
--lgs en-mk                                \ 
--ae_steps en,mk                           \
--bt_steps en-mk-en,mk-en-mk               \
--word_shuffle 3                           \ 
--word_dropout 0.1                         \
--word_blank 0.1                           \
--lambda_ae 0:1,100000:0.1,300000:0        \
--encoder_only False                       \
--emb_dim 1024                             \
--n_layers 6                               \
--n_heads 8                                \
--dropout 0.1                              \
--attention_dropout 0.1                    \
--gelu_activation true                     \
--tokens_per_batch 1000                    \
--batch_size 32                            \
--bptt 256                                 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 50000                         \
--eval_bleu true                           \
--stopping_criterion valid_mk-en_mt_bleu,10  \ 
--validation_metrics valid_mk-en_mt_bleu   \
--increase_vocab_for_lang en               \
--increase_vocab_from_lang mk              

```



## RE-LM + adapters 
###1. This step is the same as with RE-LM.

###2. Fine-tune part of the model on the target language only using adapters

```
python train.py                             \
--exp_name finetune_en_mlm_mk_adapters      \
--dump_path ./dumped/                       \
--reload_model 'mono_mlm_en_68m.pth'        \
--data_path ./data/mk-en                    \
--lgs 'en-mk'                               \
--clm_steps ''                              \
--mlm_steps 'mk'                            \
--mlm_eval_steps 'en'                       \
--emb_dim 1024                              \
--n_layers 6                                \
--n_heads 8                                 \
--dropout 0.1                               \
--attention_dropout 0.1                     \
--gelu_activation true                      \
--use_adapters True # this enables adapters and freezes Transformer layers (except embed + linear) \
--adapter_size 256                          \
--batch_size 32                             \
--bptt 256                                  \
--optimizer adam,lr=0.0001                  \
--epoch_size 50000                          \
--validation_metrics valid_mk_mlm_ppl       \
--stopping_criterion valid_mk_mlm_ppl,10    \
--increase_vocab_for_lang en                \
--increase_vocab_from_lang mk               \
--increase_vocab_by NUMBER (see ./data/mk-en/vocab.mk-en-ext-by-$NUMBER) \
```

###3. Train a UNMT model (encoder and decoder initialized with RE-LM + adapters)

```
python train.py                            \
--exp_name unsupMT_ft_mk                   \
--dump_path ./dumped/                      \
--reload_model 'finetune_en_mlm_mk_adapters.pth,finetune_en_mlm_mk_adapters.pth' \
--data_path './data/mk-en'                 \
--lgs en-mk                                \
--ae_steps en,mk                           \
--bt_steps en-mk-en,mk-en-mk               \
--word_shuffle 3                           \
--word_dropout 0.1                         \
--word_blank 0.1                           \
--lambda_ae 0:1,100000:0.1,300000:0        \
--encoder_only False                       \
--emb_dim 1024                             \
--n_layers 6                               \
--n_heads 8                                \ 
--dropout 0.1                              \
--attention_dropout 0.1                    \
--gelu_activation true                     \
--use_adapters True                        \
--adapter_size 256                         \
--tokens_per_batch 1000                    \
--batch_size 32                            \
--bptt 256                                 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 50000                         \
--eval_bleu true                           \
--stopping_criterion valid_mk-en_mt_bleu,10 \ 
--validation_metrics valid_mk-en_mt_bleu   \
--increase_vocab_for_lang en               \
--increase_vocab_from_lang mk              
```

For the XLM baseline, follow the instructions in [XLM github page](https://github.com/facebookresearch/XLM
