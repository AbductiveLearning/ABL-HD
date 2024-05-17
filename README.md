# ABL-HD

🌟 **New!** [ABLkit](https://github.com/AbductiveLearning/ABLkit) released: A toolkit for Abductive Learning with high flexibility, easy-to-use interface, and optimized performance. Welcome to try it out!🚀

# KESAR: Knowledge Enhanced Historical Document Sagmentation and Recognition

This is the code for KESAR, an Abductive Learning-based model training method for historical document segmentation and recognition.

# Publication

**"Knowledge-Enhanced Historical Document Segmentation and Recognition"**. *En-Hao Gao, Yu-Xuan Huang, Wen-Chao Hu, Xin-Hao Zhu, Wang-Zhou Dai*. In: Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI’24), Vancouver, Canada, 2024, pp.8409-8416 (Oral).[[paper](http://www.lamda.nju.edu.cn/gaoeh/paper/AAAI24-KESAR.pdf)]

# Environment

```
conda create -n kesar python=3.7
conda activate kesar
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install onnx onnxruntime-gpu python-opencv scipy scikit-image shapely tqdm segmentation_models_pytorch
```

# Inference

You need to change 3 paths before inference.
+ Segmentation model path, Line 19 ~ 22 in TestModel.py
+ Word model path, Line 30 ~ 31 in TestModel.py
+ Image path, Line 116 in TestModel.py

After changing these paths, you can run the following commond to conduct inference:

```
python TestModel.py
```

Results will be saved into the "outputs/" folder.

# Citation

```
@inproceedings{KESAR2024Gao,
  author     = {Gao, En-Hao and Huang, Yu-Xuan and Hu, Wen-Chao and Zhu, Xin-Hao and Dai, Wang-Zhou},
  title      = {Knowledge-Enhanced Historical Document Segmentation and Recognition},
  booktitle  = {Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI'24)},
  pages      = {8409--8416},
  year       = {2024}
}

```