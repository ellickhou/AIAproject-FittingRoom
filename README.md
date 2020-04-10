# AIAproject-FittingRoom
- Virtual Try-On scheme base on cycleGAN.
- Top3 project in Taiwain AI Academy finals.(Technical Professionals)
- Use cycle consistency to improve some unreasonable results either due to geometric matching limit or autoencoder-only framework.
- Code is developed and tested with Pytorch==0.4.1, torchvision==0.2.1.
- _TOC_
   - [Dataset](#Dataset)
   - [User guide](#User guide)
   - [Results](#Results)


## Dataset
- Base on [CP-VTON](https://github.com/sergeywong/cp-vton) dataset.
- Use [Self Correction for Human Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing) to achieve better parsing results.
- Add male model images extracted from [FashionGen dataset](https://fashion-gen.com) to extend real word usage.
## User guide
### Training
- example command
'''
python train.py --name gmm_train_new --stage GMM --workers 4 --save_count 5000 --shuffle
'''
   
## Results
- This project's Try-On Module refer to [Virtually Trying on New Clothing with Arbitrary Poses](https://www.english.com.tw/modules/newbb/viewtopic.php?post_id=928), but change the training strategy and dataflow for different purpose.
### 
