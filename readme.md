# SKS-Net: Skeleton-strengthening network for surface defect segmentation
## Try to make this as a productive project(on going)
- author: Yubo Zheng / Rongdi Wang
- email: 1287293308@qq.com / wangyi@s.upc.edu.cn
## Implement this paper by Pytorch
[SKS-Net](Our article has been accepted by "Measurement Science and Technology".)

## Network Arch
![network arch](doc/SKS-Net.PNG)

## usage
Taking the KolektorSDD1 dataset as an example:

1. Download the KolektorSDD1 dataset and store it in the path ``<your-path>``;
2. Modify the dataset path in the file `utils/generate_txt.py` to the location where you have stored the dataset on your computer ``<your-path>``;
3. Run the script `python utils/generate_txt.py`, which will generate the corresponding annotation files train.txt and test.txt in the location where you have stored the dataset. This will facilitate the instantiation of the data flow;
4. Finally, in the `train.py` file, change the `"--benchmark"` parameter to `"KolektorSDD1"`, and run it `python train.py` to start training and testing. During the training process, the paths `"save/"` and `"log/"` will be generated to store the relevant records.
5. Additionally, after training, you can call fuse_model from models.sksnet to switch all SkeletonStrengtheningBlock and SkeletonStrengtheningBlock2 layers to deploy mode. This fuses parallel convolutions and their BN parameters into single conv-BN pairs, yielding faster inference without changing the overall FLOPs or parameter count. Like a simple test in sksnet.py

## Currently supported datasets:
1. KolektorSDD1
2. NEU-Seg
3. CrackForest
4. RSDD1
5. RSDD2

## Currently supported models:
1. U-Net
2. A-Net
3. DeepLabV3
4. PGA-Net
5. EDR-Net
6. PSP-Net
7. ResU-Net++
8. SegFormer
9. Seg-Net
10. SKS-Net

cite bibtex style
```
@article{zuo2024sks,
  title={SKS-Net: skeleton-strengthening network for surface defect segmentation},
  author={Zuo, Haiqiang and Zheng, Yubo and Huang, Qizhou and Wang, Rongdi},
  journal={Measurement Science and Technology},
  volume={36},
  number={1},
  pages={016045},
  year={2024},
  publisher={IOP Publishing}
}
```