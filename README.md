---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/zshuai8/PDMBenchmark
cd PDMBench
```

### 2. Install Requirements

We recommend Python 3.8+ and a virtual environment. Then run:

```bash
pip install -r requirements.txt
```

---

### 3. Prepare Datasets

Download and unzip the datasets into the `./dataset/` directory. We support all standard datasets defined in the benchmark, including:

* Paderborn, CWRU, FEMTO, IMS, MFPT, XJTU, HUST Bearing, etc. Our dataset can be downloaded from https://huggingface.co/collections/odysseywt/pdmlibrary-682504064fc74fcd889cc17f

See the `dataset_mapping` inside the codebase for full descriptions.

---

### 4. Run Benchmark Experiments

To train and evaluate a model via CLI:

```bash
python run.py --task_name classification --model TimesNet --data 01 --is_training 1
```

Modify `--model`, `--data`, and other hyperparameters as needed. All models listed in TSLib (e.g., `TimeMixer`, `iTransformer`, `PatchTST`, `Koopa`, etc.) are supported.

---

## 🔗 Acknowledgment

Our benchmark framework is built upon and extends the excellent [Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library), an open-source project that provides a unified and extensible codebase for evaluating state-of-the-art deep learning models in time series tasks. We greatly appreciate the efforts of the TSLib team and leverage many of their models and utilities in our work.

If you find our benchmark useful, please also consider citing the TSLib's foundational paper:

```bibtex
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```

### 5. Use the Streamlit App (Interactive GUI)

To launch the interactive web application:

```bash
streamlit run app.py
```

This interface allows you to:

* 🧭 Browse and visualize time-domain and frequency-domain characteristics of PdM datasets
* ⚙️ Select, configure, and train SOTA models interactively
* 📊 View and compare model performance across datasets via dynamic leaderboards
* 🧪 Monitor training progress and visualize metrics in real time

---

## 📫 Contact

For questions, please open an [issue](https://github.com/zshuai8/PDMBenchmark) or reach out via [email](zshuai8@vt.edu).
