# CFIR
 # [CFIR: Fast and Effective Document-To-Image Retrieval for Large Corpora]

Official PyTorch implementation of paper: CFIR: Fast and Effective Document-To-Image Retrieval for Large Corpora.


The proposed objective can be evaluated using 4 A6000-48GB:

## Stage 1 retreival: Entity-based Ranking
```bash   
--model beit3_large_patch16_224
--input_size 224
--task 'atomic_stage1'
--batch_size 128
--sentencepiece_model 'beit3.spm'
--finetune '../path/to/your/beit3_large_itc_patch16_224.pth'
--data_path '/path/to/your/ATOMIC/dataset'
--output_dir './output/'
--log_dir './log/'
--num_workers 8
--num_max_bpe_tokens 16
--eval_batch_size 256
--eval
--retrieval_mode 'text_to_image'
--load_image_from_precomputed_npy
--produce_stage1_candidates
--use_entity
```

- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*128*1 = 1024`.
- `--finetune`: weight path of your pretrained models.
- `--task`: task to be used for this run.
- `--use_entity`: use entity as the query.
- `--load_image_from_precomputed_npy`: use the image embedding cache.
- `--produce_stage1_candidate`: produce entity candidates for next re-ranking and store in index.


## Stage 2 retreival: Summary-Based Re-ranking
```bash   
--model beit3_large_patch16_224
--input_size 224
--task 'atomic_stage2'
--batch_size 128
--layer_decay 0.65
--lr 2e-4
--epochs 30
--warmup_epochs 3
--drop_path 0.2
--sentencepiece_model 'beit3.spm'
--finetune '../path/to/your/beit3_large_itc_patch16_224.pth'
--data_path '/path/to/your/ATOMIC/dataset'
--output_dir './output/'
--log_dir './log/'
--weight_decay 0.05
--seed 42
--save_ckpt_freq 1
--num_workers 8
--num_max_bpe_tokens 64
--eval_batch_size 256
--retrieval_mode 'text_to_image'
--load_image_from_precomputed_npy
--produce_stage2_candidates
--use_summary
```
- `--use_sumamry`: use summary as the query.
- `--load_image_from_precomputed_npy`: use the image embedding cache.
- `--load_stage1_candidates`: use stage1 candidates as the retrieval set.







## Citation

If you find this repository useful, please consider citing works:
```

```



## Acknowledgement

This repository is built using the [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3) repository and the [timm](https://github.com/rwightman/pytorch-image-models) library.



