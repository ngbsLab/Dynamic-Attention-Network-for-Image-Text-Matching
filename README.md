## Intriduction
This is Danamic Attention Network, source code of DMAN. It is built on top of the SCAN in PyTorch.
![image01](https://user-images.githubusercontent.com/48584373/106551680-1abcaf00-6559-11eb-914d-d643f49d32b5.png)
## Requirements and Installation
We recommended the following dependencies.
* Python2.7
* [PyTorch](https://pytorch.org/) 1.4.0
* [Numpy](https://numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

## Download data
Download the dataset files. We use the image feature created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN).   
Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](https://cs.stanford.edu/people/karpathy/deepimagesent/).   
The precomputed image features of MS-COCO are from [here](https://github.com/peteanderson80/bottom-up-attention). The precomputed image features of Flickr30K are extracted from the raw Flickr30K images using the bottom-up attention model from [here](https://github.com/peteanderson80/bottom-up-attention). All the data needed for reproducing the experiments in the paper, including image features and vocabularies, can be downloaded from:
```
wget https://scanproject.blob.core.windows.net/scan-data/data.zip
wget https://scanproject.blob.core.windows.net/scan-data/vocab.zip
```
We refer to the path of extracted files for data.zip as $DATA_PATH and files for vocab.zip to ./vocab directory. Alternatively, you can also run vocab.py to produce vocabulary files. For example,
```
python vocab.py --data_path data --data_name f30k_precomp
python vocab.py --data_path data --data_name coco_precomp
```
## Training new models
Run train.py:
```
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/coco_scan/log --model_name runs/coco_scan/log --max_violation --bi_gru
```

## Evaluate trained models
```
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$RUN_PATH/coco_scan/model_best.pth.tar", data_path="$DATA_PATH", split="test")
```
To do cross-validation on MSCOCO, pass fold5=True with a model trained using --data_name coco_precomp.
