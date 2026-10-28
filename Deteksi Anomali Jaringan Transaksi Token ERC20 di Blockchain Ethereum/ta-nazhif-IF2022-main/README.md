# ta-nazhif-IF2022

# 🔬 Sistem Deteksi Anomali Blockchain dengan GraphSAGE

Sistem deteksi anomali berbasis Graph Neural Network (GNN) untuk mengidentifikasi transaksi mencurigakan pada blockchain Ethereum.

## 📁 Struktur Proyek

```
.
├── app.py                  # Main Streamlit application
├── config.py               # Konfigurasi dan konstanta
├── model.py                # Definisi model GraphSAGE
├── data_loader.py          # Fungsi loading dan preprocessing data
├── trainer.py              # Fungsi training dan evaluasi
├── visualizations.py       # Fungsi visualisasi
├── requirements.txt        # Dependencies
├── README.md              # Dokumentasi
└── transaksiERC20.csv     # Dataset (opsional)
```

## 🎯 Fitur Utama

### 1. **config.py**
- Konfigurasi default model
- Definisi label kelas (Normal, Ice Phishing, NFT Scam, Reward Scam)
- Path dataset default

### 2. **model.py**
- Implementasi GraphSAGE dengan PyTorch Geometric
- Multi-layer architecture dengan Batch Normalization
- Fungsi untuk mendapatkan node embeddings

### 3. **data_loader.py**
- Loading dataset dari CSV
- Preprocessing data transaksi
- Feature engineering (in-degree, out-degree, PageRank, dll)
- Graph construction menggunakan NetworkX
- Data splitting (Train/Val/Test)

### 4. **trainer.py**
- Training loop dengan early stopping
- Multiple runs evaluation
- Per-label metrics calculation
- Real-time visualization training progress

### 5. **visualizations.py**
- Overview dataset
- Distribusi label kelas
- Statistik graph
- Confusion matrix
- Classification report
- t-SNE visualization
- Graph network visualization
- Detection results comparison

### 6. **app.py**
- Main Streamlit interface
- Parameter configuration sidebar
- Workflow orchestration

## 🚀 Cara Menggunakan

### Instalasi

```bash
# Clone repository atau download files
# Install dependencies
pip install -r requirements.txt
```

### Menjalankan Aplikasi

```bash
streamlit run app.py
```

### Workflow

1. **Upload Dataset** atau gunakan dataset default
2. **Konfigurasi Parameter** di sidebar:
   - Hidden Channels
   - Number of Layers
   - Dropout Rate
   - Learning Rate
   - Epochs
   - dll
3. **Klik "Mulai Proses Training & Evaluasi"**
4. **Lihat Hasil**:
   - Graph statistics
   - Training progress
   - Confusion matrix
   - Classification report
   - t-SNE visualization
   - Detection results

## 📊 Output

- **Training Metrics**: Loss dan F1 score per epoch
- **Evaluation Results**: Accuracy, F1-score (macro & weighted)
- **Per-Label Metrics**: Precision, Recall, F1-score untuk setiap kelas
- **Visualizations**: 
  - Confusion matrix
  - t-SNE projection
  - Graph network structure
  - Detection comparison table

## ⚙️ Konfigurasi Default

```python
DEFAULT_CONFIG = {
    'hidden_channels': 128,
    'num_layers': 3,
    'dropout': 0.5,
    'epochs': 200,
    'learning_rate': 0.005,
    'weight_decay': 5e-4,
    'patience': 15
}
```

## 📝 Format Dataset

Dataset harus berupa CSV dengan kolom minimal:
- `From`: Alamat pengirim
- `To`: Alamat penerima
- `Value (USD)`: Nilai transaksi
- `Amount`: Jumlah token
- `Method`: Metode transaksi
- `From_NameTag`: Tag pengirim
- `To_NameTag`: Tag penerima
- `label`: Label kelas (0-3)

## 🔧 Customization

### Menambah Label Baru
Edit `config.py`:
```python
LABEL_NAMES = {
    0: 'Normal',
    1: 'Ice Phishing',
    2: 'NFT Scam',
    3: 'Reward Scam',
    4: 'Your New Label'  # Tambahkan di sini
}
```

### Mengubah Arsitektur Model
Edit `model.py` untuk memodifikasi GraphSAGE architecture

### Menambah Visualisasi
Tambahkan fungsi baru di `visualizations.py`

## 📦 Dependencies

- streamlit
- pandas
- numpy
- torch
- torch-geometric
- networkx
- matplotlib
- seaborn
- scikit-learn
- joblib

## 🎓 Metodologi

1. **Feature Engineering**: 
   - Graph features (degree, PageRank)
   - Transaction features (value, method)
   - Tag-based features

2. **Graph Construction**:
   - Nodes: Unique addresses
   - Edges: Transactions

3. **Model**: GraphSAGE
   - Aggregate neighborhood information
   - Multi-layer architecture
   - Class-weighted loss untuk imbalanced data

4. **Evaluation**:
   - Multiple runs dengan random seeds
   - Best model selection berdasarkan F1-score
   - Per-label performance analysis

## ⚠️ Catatan

- Pastikan dataset memiliki kolom yang sesuai
- Untuk dataset besar, proses preprocessing mungkin memakan waktu
- GPU direkomendasikan untuk training yang lebih cepat
- Multiple runs akan meningkatkan waktu eksekusi
