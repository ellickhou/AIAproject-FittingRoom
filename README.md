# AIAproject-FittingRoom
- Virtual Try-On scheme base on cycleGAN.
- Use cycle consistency to fix the unreasonable result either due to geometric matching limit or autoencoder-only framework.
- Code is developed and tested with Pytorch==0.4.1, torchvision==0.2.1.
- Top3 project in Taiwain AI Academy finals.(Technical Professionals)
_TOC_
* [Dataset](#Dataset)
    * [Synthesis](#synthesis)
    * [Classification](#classification)
    * [Recommendation](#recommendation)
    * [Forecast](#forecast)
* [Related Events](#related-events)
* [Datasets](#datasets)
* [Companies](#companies)
* [Other Useful Resources](#other-useful-resources)

## Dataset
- Base on [CP-VTON](https://github.com/sergeywong/cp-vton) dataset.
- Use [Self Correction for Human Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing) to achieve better parsing results.
- Add male model images extracted from [FashionGen dataset](https://fashion-gen.com) to extend real word usage.
