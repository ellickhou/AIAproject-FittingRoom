# AIAproject-FittingRoom
<div align="center">
 <img src="image/result.png" width="500px" />
 <p>fig1. Testing result for different Try-On task.</p>
</div>

- Virtual Try-On scheme base on cycleGAN.
- Top3 project in Taiwain AI Academy finals.(Technical Professionals)
- Use cycle consistency to improve some unreasonable results either due to geometric matching limit or autoencoder-only framework.
- Code is developed and tested with Pytorch==0.4.1, torchvision==0.2.1.
- _TOC_
   - [Dataset](#Dataset)
   - [User_guide](#User_guide)
     - [Training](#Training)
     - [Testing](#Testing)
   - [Discussion](#Discussion)
      - [Parsing_influence](#Parsing_influence)


## Dataset
- Base on [CP-VTON](https://github.com/sergeywong/cp-vton) dataset.
- Use [Self Correction for Human Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing) to achieve better parsing results.
- Add male model images extracted from [FashionGen dataset](https://fashion-gen.com) to extend real word usage.
## User_guide
- This project's Try-On Module refer to [Virtually Trying on New Clothing with Arbitrary Poses](https://www.english.com.tw/modules/newbb/viewtopic.php?post_id=928), but change the training strategy and dataflow for different purpose.
### Training
- example command, ```--uselimbs``` is an option for certain Try-On task, please see
```
python train_cycleTryOn.py --name 'gmm_train' --stage 'GMM' --save_count 5000 --shuffle
python train_cycleTryOn.py --name 'cycleTryOn_train' --stage 'cycleTryOn' --uselimbs
```
### Testing
```
python test_cycleTryOn.py --name 'gmm_test' --stage 'GMM' --datamode test --data_list 'test_pairs.txt' --checkpoint checkpoints/gmm_train/gmm_final.pth
python test_cycleTryOn.py --name 'cycleTryOn_test' --stage 'cycleTryOn' --uselimbs --datamode test --data_list 'test_pairs.txt' --checkpoint checkpoints/cycleTryOn_train/cycleTryOn_final.pth
```
## Discussion
### 
### more
#### Parsing_influence
<div align="center">
 <img src="image/d1.png" width="203px" />
 <p>fig2. Higher IoU dose have better GMM result.</p>
</div>
