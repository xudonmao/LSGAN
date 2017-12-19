# LSGAN
Code for [Least Squares Generative Adversarial Networks](http://openaccess.thecvf.com/content_iccv_2017/html/Mao_Least_Squares_Generative_ICCV_2017_paper.html) and [On the Effectiveness of Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1712.06391).

Related project: [improved_LSGAN](https://github.com/xudonmao/improved_LSGAN)
### Prerequisites
- Tensorflow
- SciPy

### Usage
1.Download datasets
  - [LSUN](http://lsun.cs.princeton.edu/2016/)
  - [HWDB1.0 (Handwritten Chinese characters)](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html)

2.Convert data

```
  $ python convert_lsun.py --source_dir $SOURCE_PATH --target_dir $TARGET_PATH
  $ python convert_icdar.py --source_dir $SOURCE_PATH --target_dir $TARGET_PATH
```

3.Train model

```
  $ python train_lsun.py --data_path $DATA_PATH
  $ python train_chn.py --data_dir $DATA_DIR
```

### Citation
If you use this work in your research, please cite:

    @article{arxiv1611.04076,
      author = {Xudong Mao and Qing Li and Haoran Xie and Raymond Y.K. Lau and Zhen Wang and Stephen Paul Smolley},
      title = {Least Squares Generative Adversarial Networks},
      journal = {arXiv preprint arXiv:1611.04076},
      year = {2016}
    }
    
 
### Results
#### Church
 ![result](images/church_supplementary.png)
#### Bedroom
 ![result](images/bedroom_supplementary.png)
#### Kitchen
 ![result](images/kitchen_supplementary.png)
#### Conference
 ![result](images/conference_supplementary.png)
#### Dining
 ![result](images/dining_supplementary.png)
