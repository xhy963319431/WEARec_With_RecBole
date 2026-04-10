# WEARec_With_RecBole (AAAI'26 Oral)
This is the official repository for our paper "**Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation**" in AAAI'26 Oral.
You can find our paper on [[arXiv][(https://arxiv.org/abs/2511.07028)].

## 🛠️ Installation
To replicate our environment, please follow the steps below:
```bash
# Clone this repository
git clone https://github.com/HyeYoung1218/DIFF.git
cd DIFF

# Create conda environment from the provided file
conda env create -f environment.yml
conda activate diff
```

## 📁 Dataset
We utilize four preprocessed datasets:

* `Amazon_Beauty`
* `Amazon_Sports_and_Outdoors`
* `Amazon_Toys_and_Games`
* `yelp`

Please download the datasets manually from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or the provided [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI) links, and place them under the `dataset/` directory:

```
dataset/
├── Amazon_Beauty/
├── Amazon_Sports_and_Outdoors/
├── Amazon_Toys_and_Games/
└── yelp/
```

Each dataset corresponds to a configuration file located in the `configs/` folder. These `.yaml` files define dataset-specific training settings such as model parameters, data paths, and training schedule.

```
configs/
├── Amazon_Beauty_WEARec.yaml
├── Amazon_Sports_and_Outdoors_WEARec.yaml
├── Amazon_Toys_and_Games_WEARec.yaml
└── yelp_WEARec.yaml
```

## 🚀 Reproduction
To train and evaluate our model, simply run the shell script `run_wearec.sh`, which includes predefined training commands for each dataset. This script contains the best-performing hyperparameters for each dataset.

Run the following command:
```bash
bash run_wearec.sh
```

## 📄 Citation
If you find this repository helpful for your work, please cite the following paper:

```bibtex
@inproceedings{xu2026wavelet,
  title={Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation},
  author={Xu, Huayang and Yuan, Huanhuan and Liu, Guanfeng and Fang, Junhua and Zhao, Lei and Zhao, Pengpeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={19},
  pages={16058--16065},
  year={2026}
}
```

## 🔗 Acknowledgement
This repository is based on [RecBole](https://github.com/RUCAIBox/RecBole) and [DIFF](https://github.com/HyeYoung1218/DIFF).
