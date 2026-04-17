<h1 align="center">Multimodal Perception and Three-Dimensional Force Decoupling Method for a Multidimensional Bionic Tactile Sensor</h1>
<div align=center>

 <p align="center">Dapeng Chen, Peng Gao, Zhenjie Ma, Zhou Zhuang, Zhangjia Deng, Hui Zhang, Yun Ling, Xuhui Hu, Hong Zeng, Lina Wei, Jia Liu, Qiang Zhao, Aiguo Song</p>
  <p align="center">Nanjing University of Information Science & Technology</p>
  ---
</div>
  
  Sensor model, fabrication method, control code, and model code. The following two images show the structure and prototype of our sensor.
<div align=center>
<img src="https://github.com/AILM-UX/Multidimensional-Tactile-Sensor/main/figure/fig1a.jpg" alt="Image text" width="250" height="200"/>     <img src="https://github.com/AILM-UX/Multidimensional-Tactile-Sensor/main/figure/fig1b.jpg" alt="Image text" width="250" height="200"/>
</div>

In this work, the framework we propose consists of two main component. The first part is the ST attention-based VTDF module, which enables organic interaction and fusion of input visual and tactile data, generating fused visual-tactile features that are fully interacted in terms of inter-modal and ST information. The second part is the cross-task attention-driven MTL module. The fused visual-tactile features obtained from the first part serve as shared features within the MTL module, facilitating inter-task information interaction and fusion to output recognition results for multiple object attributes.

![image](https://github.com/AILM-UX/drwxx/raw/main/Framework%20diagram%20of%20an%20attribute%20recognitionmethod.png)
## <p align="center">DETAILS OF IMPLEMENT</p>
This project includes scripts for requirements, model training and testing.

### Requirements

- Python 3.12
- torch 2.7.0
- CUDA 11.8

### Dataset
We use the TVL dataset as the test and validation set for the object attribute recognition framework. The TVL dataset consists of the SSVTP dataset and the HCT dataset. The visual input is RGB images from a Logitech BRIO webcam, and the tactile input is tactile images from DIGIT. The SSVTP dataset comprises 4,587 pairs of visual and tactile images, while the HCT dataset comprises 39,154 pairs. For each pair of data in SSVTP, staff manually annotated the attribute information. For each pair of data in the HCT dataset, GPT-4V annotated the attributes, resulting in a dataset containing 43,741 pairs of visual-tactile images.
For more details, see https://arxiv.org/abs/2402.13232

### Data Preparation
Modify the dataset path in `dataloador.py`
```bash
python dataloador.py
```
### Model Training
Use `MQTransformer_Uncertainty Loss_train.py` to train the model. Adjust parameters such as epoch and batch_size based on the training environment.
```bash
python MQTransformer_Uncertainty Loss_train.py
```
After training, the model will be saved in the specified directory.

## Testing
Use `result.py` to validate results on the test set.
```bash
python result.py
```

