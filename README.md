# [CVPR 2024]. Efficient Meshflow and Optical Flow Estimation from Event Cameras. [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_Efficient_Meshflow_and_Optical_Flow_Estimation_from_Event_Cameras_CVPR_2024_paper.pdf).
<h4 align="center">Xionglong Luo<sup>1,4</sup>, Ao Luo<sup>2,4</sup>, Zhengning Wang<sup>1</sup>, Chunyu Lin<sup>3</sup>, Bing Zengn<sup>1</sup>, Shuaicheng Liu<sup>1,4</sup></center>
<h4 align="center">1.University of Electronic Science and Technology of China
<h4 align="center">2.Southwest Jiaotong University, 3.Beijing Jiaotong University, 4.Megvii Technology </center></center>
  
## Environments
You will have to choose cudatoolkit version to match your compute environment. The code is tested on Python 3.7 and PyTorch 1.10.1+cu113 but other versions might also work. 
```bash
conda create -n EEMFlow python=3.7
conda activate EEMFlow
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements
```
## Dataset
### MVSEC
You need download the HDF5 files version of [MVSEC](https://daniilidis-group.github.io/mvsec/download/) datasets. We provide the code to encode the events and flow label of MVSEC dataset.
```python
# Encoding Events and flow label in dt1 setting
python loader/MVSEC_encoder.py --only_event -dt=1
# Encoding Events and flow label in dt4 setting
python loader/MVSEC_encoder.py --only_event -dt=4
# Encoding only Events
python loader/MVSEC_encoder.py --only_event
```

### HREM
This work proposed a Multi Density Rendered (HREM) event optical flow dataset, you can download it from https://pan.baidu.com/s/1iSgGCjDask-M_QqPRtaLhA?pwd=z52j . We also provide code for batch organizing HREM datasets.

## Evaluate
### Pretrained Weights
Pretrained weights can be downloaded from 
[Google Drive](https://drive.google.com/drive/folders/15uwhrmUzg3kK3UB6z0Qnht-sGs7Nq23o?usp=sharing).
Please put them into the `checkpoint` folder.

### Test on HREM
```python
python test_EEMFlow_HREM.py -dt dt1
python test_EEMFlow_HREM.py -dt dt4
```
## Citation

If this work is helpful to you, please cite:

```
@inproceedings{luo2024efficient,
  title={Efficient Meshflow and Optical Flow Estimation from Event Cameras},
  author={Luo, Xinglong and Luo, Ao and Wang, Zhengning and Lin, Chunyu and Zeng, Bing and Liu, Shuaicheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19198--19207},
  year={2024}
}
```
## Acknowledgments

Thanks the assiciate editor and the reviewers for their comments, which is very helpful to improve our paper. 

Thanks for the following helpful open source projects:

[ERAFT](https://github.com/uzh-rpg/E-RAFT),
[TMA](https://github.com/ispc-lab/TMA),
[ADMFlow](https://github.com/boomluo02/ADMFlow/).

