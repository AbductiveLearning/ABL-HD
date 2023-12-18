### Model Weights
Segmentation Model：
+ MTH: https://drive.google.com/file/d/1jTCs-nk2mtONtPJr9pY2gtqsV80zIS70/view?usp=drive_link
+ GBACHD: https://drive.google.com/file/d/1haONSHvMe1eShWpGQ-ZO_OzOn21HG1bj/view?usp=drive_link

Word Model
+ MTH: https://drive.google.com/file/d/1evSm1jcXLbcGF4DXVvNy4BoPPvP8eoCG/view?usp=drive_link
+ GBACHD: https://drive.google.com/file/d/1nshzx30_hzrvi0JyrW_FbLVxVuNuiQSh/view?usp=drive_link

All weights should be placed under the "weights/" folder.

### Environment

```
conda create -n inference-nine-floor python=3.7
conda activate inference-nine-floor
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install onnx onnxruntime-gpu python-opencv scipy scikit-image shapely tqdm segmentation_models_pytorch
```

### Inference

You need to change 3 paths before inference.
+ Segmentation model path, Line 19 ~ 22 in TestModel.py
+ Word model path, Line 30 ~ 31 in TestModel.py
+ Image path, Line 116 in TestModel.py

After changing these paths, you can run the following commond to conduct inference:

```
python TestModel.py
```

Results will be saved into the "outputs/" folder.