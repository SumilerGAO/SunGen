# SunGen

This repository contains the code for our paper [“**SunGen: Self-Guided High-Quality Data Generation in Efficient Zero-Shot Learning**”](https://arxiv.org/abs/2202.07922). 

**Data generation**

For data generation via PLM, the implementation is built on the source code from [ZeroGen](https://github.com/jiacheng-ye/ZeroGen). For movie review sentiment classification tasks (imdb, sst-2, rotten tomato), we use the same prompts as ZeroGen. For other tasks, we provide the detailed prompts for each task in this repository under `./tasks/`.  

We provide sample codes for yelp data generation:

(1) generate restaurant name

```python
python main.py --reload_model  --task_file tasks/yelp/yelp-x1.json --input_file_type plain --output_dir yelp/output/yelp-x1-gen/ --model_name  gpt2-xl --small_model_name distilbert -base-uncased  --min_length 1 --max_length 5  --top_k 0 --top_p 0.9 --decay_constant 200 --batch_size 2048  --train_batch_size 32 --learning_rate 2e-5 --num_entries_per_input 500000
```

(2) generate  restaurant review dataes given restaurant name

```python
python main.py --reload_model  --task_file tasks/yelp/yelp-x2.json --output_dir   yelp/output/yelp-x1/ --input_file_type 'plain' --input_file tasks/subj/res_names.txt --model_name  gpt2-xl --small_model_name distilbert-base-uncased  --min_length 10 --max_length 100  --top_k 0 --top_p 0.9 --decay_constant 200 --batch_size 180  --train_batch_size 32 --learning_rate 2e-5 --num_entries_per_input 1000000
```

More details can be found on our paper. 

**Run with generated data**

After dataset generation, we save the synthetic dataset at `train.jsonl`. The file is in json line format (e.g., `{"idx": 0, "text": "The Book of Mormon Musical brings all the drama and excitement of a real revival of the Broadway production to the big screen.", "label": 0}`).  We provide some sample synthetic set and standard sets in this [google drive link](https://drive.google.com/file/d/1jvnTObeUSZylmkWjwDWQDr4Qb2J9FBkn/view?usp=sharing).

To learn the sample reweighs using LSTM as TAM, please use the following script. 

```python
python run_reweight.py --gold_data_path data/imdb/std/ --syn_data_path data/imdb/gpt2-xl/ --task_name imdb --num_use_samples_inner 1000000 --num_use_samples_outer 50000 --epoch_converge 1 --outer_lr 2.5e-1 --inner_lr 1e-3 --seed 12345 --backward_batch_size 4096 --wandb --outer_obj combined --inner_obj ce --init_label 10 --theta_upper_lim 1 --check_ft_every 5 --epoch_converge_fully_train 5 --threshold 0.9 --optim Adam --max_outer_iter 100 --hard --init_theta 1 --subset_outer --use_sigmoid --disable_outer_scheduler --shuffle_train

```

**Acknowledgement**

If you find our code useful, please cite our paper:

```html
@inproceedings{
gao2023selfguided,
title={Self-Guided Noise-Free Data Generation for Efficient Zero-Shot Learning},
author={Jiahui Gao and Renjie Pi and LIN Yong and Hang Xu and Jiacheng Ye and Zhiyong Wu and WEIZHONG ZHANG and Xiaodan Liang and Zhenguo Li and Lingpeng Kong},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=h5OpjGd_lo6}
}

@inproceedings{ye-etal-2022-progen,
title = "ProGen: Progressive Zero-shot Dataset Generation via In-context Feedback",
author = "Ye, Jiacheng and Gao, Jiahui and Wu, Zhiyong and Feng, Jiangtao and Yu,Tao     		and Kong, Lingpeng",
booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
month = dec,
year = "2022",
address = "Abu Dhabi, United Arab Emirates",
publisher = "Association for Computational Linguistics",
url = "https://aclanthology.org/2022.findings-emnlp.269",
pages = "3671--3683"
}

@inproceedings{ye-etal-2022-zerogen,
title = "{Z}ero{G}en: Efficient Zero-shot Learning via Dataset Generation",
author = "Ye, Jiacheng  and Gao, Jiahui  and Li, Qintong  and Xu, Hang  and Feng, Jiangtao  and Wu, Zhiyong  and Yu, Tao  and Kong, Lingpeng",
booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
year = "2022"
}
```

