## Ought to be Salient and Hidden:Soft Biological Semantic-Guided Explicit-Implicit Learning for Cloth-Changing Person Re-Identification
### Journal: The Visual Computer

### Requirements
- Python 3.6
- Pytorch 1.6.0
- yacs
- apex

### Dataset

The relevant dataset can be downloaded through the following link
- LTCC is available at [Here](https://naiq.github.io/LTCC_Perosn_ReID.html)
- PRCC is available at [Here](https://drive.google.com/file/d/1yTYawRm4ap3M-j0PjLQJ--xmZHseFDLz/view)
### Get Started

Replace `_C.DATA.ROOT` and `_C.OUTPUT` in `configs/default_img.py`with your own `data path` and `output path`, respectively.

According to different datasets, run the following commands on the console to start training:

For LTCC:
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --gpu 0,1 
```
For PRCC:
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 0,1 
```
For VC-Clothes:
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset vcclothes --cfg configs/res50_cels_cal.yaml --gpu 0,1 
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset vcclothes_cc --cfg configs/res50_cels_cal.yaml --gpu 0,1 --eval --resume /Your Dataset Path/Your Model Path/ 
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset vcclothes_sc --cfg configs/res50_cels_cal.yaml --gpu 0,1 --eval --resume /Your Dataset Path/Your Model Path/ 
```
### Test

you can simply test it with
```
python test.py
```

### Citation

If you find this code useful for your research, please consider citing our paper

### Acknowledgement

Some related work can be found from the following link
- [fast-reid](https://github.com/JDAI-CV/fast-reid)
- [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
- [Pytorch ReID](https://github.com/layumi/Person_reID_baseline_pytorch)








