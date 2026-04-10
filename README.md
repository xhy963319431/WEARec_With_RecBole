# WEARec_With_RecBole (AAAI'26 Oral)
This is the official repository for our paper "**Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation**" in AAAI'26 Oral.
You can find our paper on [arXiv](https://arxiv.org/abs/2511.07028).

## 🛠️ Installation
To replicate our environment, please follow the steps below:
```bash
# Clone this repository
git clone https://github.com/xhy963319431/WEARec_With_RecBole.git
cd WEARec_With_RecBole

# Create conda environment from the provided file
conda env create -f environment.yml
conda activate wearec
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
## Experiment Results Update
We noticed that [DIFF](https://arxiv.org/abs/2510.25259) (SIGIR 2025) evaluated [BSARec](https://arxiv.org/abs/2312.10325) (AAAI 2024) on the Amazon and Yelp datasets to assess its performance within the RecBole framework. To address this, we present the corresponding performance benchmarks for WEARec below. Please note that BSARec and WEARec do not use side information here.


<table>
  <tr>
    <th rowspan="2">Metric</th>
    <th colspan="3">Yelp</th>
    <th colspan="3">Beauty</th>
    <th colspan="3">Sports</th>
    <th colspan="3">Toys</th>
  </tr>
  <tr>
    <th>BSARec</th>
    <th>WEARec</th>
    <th>DIFF</th>
    <th>BSARec</th>
    <th>WEARec</th>
    <th>DIFF</th>
    <th>BSARec</th>
    <th>WEARec</th>
    <th>DIFF</th>
    <th>BSARec</th>
    <th>WEARec</th>
    <th>DIFF</th>
  </tr>
  <tr>
    <td>HR@10</td>
    <td>0.0701</td>
    <td>0.0831</td>
    <td>0.0815</td>
    <td>0.0871</td>
    <td>0.0879</td>
    <td>0.0935</td>
    <td>0.0506</td>
    <td>0.0513</td>
    <td>0.0574</td>
    <td>0.0928</td>
    <td>0.0951</td>
    <td>0.1023</td>
  </tr>
   <tr>
    <td>HR@20</td>
    <td>0.1023</td>
    <td>0.1176</td>
    <td>0.1200</td>
    <td>0.1260</td>
    <td>0.1257</td>
    <td>0.1347</td>
    <td>0.0741</td>
    <td>0.0764</td>
    <td>0.0853</td>
    <td>0.1293</td>
    <td>0.1310</td>
    <td>0.1425</td>
  </tr>
  <tr>
    <td>NDCG@10</td>
    <td>0.0423</td>
    <td>0.0473</td>
    <td>0.0470</td>
    <td>0.0437</td>
    <td>0.0426</td>
    <td>0.0526</td>
    <td>0.0239</td>
    <td>0.0232</td>
    <td>0.0310</td>
    <td>0.0460</td>
    <td>0.0448</td>
    <td>0.0553</td>
  </tr>
  <tr>
    <td>NDCG@20</td>
    <td>0.0503</td>
    <td>0.0560</td>
    <td>0.0567</td>
    <td>0.0535</td>
    <td>0.0521</td>
    <td>0.0632</td>
    <td>0.0298</td>
    <td>0.0295</td>
    <td>0.0381</td>
    <td>0.0552</td>
    <td>0.0539</td>
    <td>0.0656</td>
  </tr>
</table>

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
