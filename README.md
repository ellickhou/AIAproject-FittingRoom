# AIAproject-FittingRoom
<div align="center">
 <img src="image/result.png" width="700px" />
 <p>Fig.1 Testing result for different type of Try-On task.</p>
</div>

- Virtual Try-On scheme base on cycleGAN.
- Top3 project in Taiwain AI Academy finals.(Technical Professionals Program)
- Use cycle consistency to improve unreasonable results either due to geometric matching or autoencoder-only framework.
- Code is developed and tested with pytorch==0.4.1, torchvision==0.2.1.
- _TOC_
   - [Dataset](#Dataset)
   - [User_Guide](#User_Guide)
     - [Training](#Training)
     - [Testing](#Testing)
   - [Discussion](#Discussion)
     - [Geometric Matching Limitation](#Geometric-Matching-Limitation)
     - [Parsing Influence](#Parsing-Influence)


## Dataset
- Base on [CP-VTON](https://github.com/sergeywong/cp-vton) dataset.
- Use [Self Correction for Human Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing) to achieve better parsing results.
- Add male model images extracted from [FashionGen dataset](https://fashion-gen.com) to extend real word usage.
## User_Guide
- The framework of Try-On Module in this project refer to [Virtually Trying on New Clothing with Arbitrary Poses](https://www.english.com.tw/modules/newbb/viewtopic.php?post_id=928), but change the training strategy and pipeline for different purpose.
### Training
- example command, ```--uselimbs``` is an option for certain Try-On task, please see [Discussion](#Discussion).
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
- In this section, we reproduce [CP-VTON](https://github.com/sergeywong/cp-vton)'s Try-On Module by the implementation details their paper provided, and compare to our model.
### Geometric Matching Limitation
- Geometric Matching Module(GMM) proposed by CP-VTON have proven its efficiency in aligning in-shop cloth with the person image. But GMM does not have the ability to tell the difference between inner side of the cloth and the outer side, in other words GMM tends to force the WHOLE in-shop cloth into the original cloth shape on person if the deformation grid is dense enough.
- In this work, cycle-consistency loss and adversarial loss are introduce to adjust this unrealistic result.
- Compared with CP-VTON, our cycleTryOn module have learned how to hide the inner part of the cloth and generate the appropriate skin color of the person as shown in Fig.2.
<div align="center">
 <img src="image/GML.png" width="700px" />
 <p>Fig.2 The lining and tag of target cloth have been warped together by geometric matching. still exist in CP-VTON </p>
</div>

### More
#### Parsing Influence
<div align="center">
 <img src="image/PI.png" width="203px" />
 <p>fig.? Higher IoU dose improve GMM result .</p>
</div>
