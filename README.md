# IEEE_TGRS_DHP-DARN

## [Hyperspectral Pansharpening Using Deep Prior and Dual Attention Residual Network](https://ieeexplore.ieee.org/document/9076645)

**Python implementation of our proposed DHP-DARN method for hyperspectral pansharpening.**

![Flowchart](https://github.com/yxzheng24/IEEE_TGRS_DHP-DARN/blob/main/Flowchart_TGRS20.png "Overall flowchart of the proposed DHP-DARN method for HS pansharpening.")

## Usage
Here we take the experiments conducted on the [Pavia Center](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_scene) data set as an example for illustration.

*   Training:
1.   Download the [Pavia Center](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_scene) scene, partition the top-left 960 × 640 × 102 part into 24 cubic-patches of size 160 × 160 × 102 with no overlap, and put the subimages into the __./data_process/pavia_subimgs/__ folder. Note that you can also download the Pavia Center data set (subimages in .mat format) from the Baidu Cloud links: https://pan.baidu.com/s/1QBYLFZpS5VHnx1A3Dkz4Dg (Access Code: prz6).
2.   Run *"get_LRHR_pavia.m"* to obtain HR-HSI, LR-HSI, bicubic upsampled HSI and nearest upsampled HSI.
3.   Run *"DHP_SR_pavia.py"* in the __./data_process/DHP_SR/__ folder to upsample the LR-HSI without learning from large data sets. Note that you can also download the upsampled HSI of Pavia Center data set via DHP method (DHP_SR_results in .mat format) from the Baidu Cloud links: https://pan.baidu.com/s/1LAUL-PJWskGJ5fOOsjrwug (Access Code: g0w4).
4.   Run *"get_cat_predLHSpaPAN.m"* to concatenate the upsampled HSI and the HR-PAN image along the spectral dimension as input, and to genertate the residual HSI simultaneously.
5.   Randomly select 17 HSI pairs from __./data_process/cat_LHS_PAN/I_predLHSpa_Pan/__ and __./data_process/Respavia_lhsconv/__ folders to form the training set.
6.   Run *"get_trainh5_pavia.m"* to produce the HDF5 file for training.
7.   Run *"turntotrain_h5.py"* first and then run *"train_DARN.py"* for training.

## Citation
Y. Zheng, J. Li, Y. Li, J. Guo, X. Wu and J. Chanussot, "Hyperspectral Pansharpening Using Deep Prior and Dual Attention Residual Network," in IEEE Transactions on Geoscience and Remote Sensing, vol. 58, no. 11, pp. 8059-8076, Nov. 2020, doi: 10.1109/TGRS.2020.2986313.

    @ARTICLE{Zheng2020TGRS,
    author={Y. {Zheng} and J. {Li} and Y. {Li} and J. {Guo} and X. {Wu} and J. {Chanussot}},
    journal={IEEE Trans. Geosci. Remote Sens.}, 
    title={Hyperspectral Pansharpening Using Deep Prior and Dual Attention Residual Network}, 
    year={2020},
    volume={58},
    number={11},
    pages={8059-8076},
    doi={10.1109/TGRS.2020.2986313}}


## Contact Information
If you have any problem, please do not hesitate to contact Yuxuan Zheng (e-mail: yxzheng24@163.com).

Yuxuan Zheng is with the State Key Laboratory of Integrated Services Networks, School of Telecommunications Engineering, Xidian University, Xi’an 710071, China.
