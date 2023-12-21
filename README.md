# openset_attribution_synthetic_images
This repo holds code for [A Siamese-based Verification System for Open-set Architecture Attribution of SyntheticImages](https://arxiv.org/pdf/2307.09822.pdf)

## Usage

### 1. Download trained models for configuration 1 and 3 (all models mentioned in the paper, are available upon request)
* [Get models from this link](https://drive.google.com/drive/folders/1b8xk1KI1xIgLvtOiwCjSvrePrQoPWTG6?usp=sharing)


### 2. Environment

Please prepare an environment with python=3.10, and then use the command "pip install -r requirements.txt" for the dependencies.

### 3. Train/Test

- Run the train script first for the embeddings then for the dense layers providing as an argument a configuration file that contains the details about the classes and their paths
- make sure when training the dense layers to pass the path of the trained embeddings

```bash
CUDA_VISIBLE_DEVICES=0 python train_siamese_embeddings.py

CUDA_VISIBLE_DEVICES=0 python train_denselayer.py
```

- Run the test script either in verification scenario or classification scenario also passing the configuration argument.

```bash
python test_denselayer.py
python test_denselayer_sota.py # this turns the verification system into classification
```


## Citations

```bibtex
@article{abady2023siamese,
  title={A Siamese-based Verification System for Open-set Architecture Attribution of Synthetic Images},
  author={Abady, Lydia and Wang, Jun and Tondi, Benedetta and Barni, Mauro},
  journal={arXiv preprint arXiv:2307.09822},
  year={2023}
}
```
