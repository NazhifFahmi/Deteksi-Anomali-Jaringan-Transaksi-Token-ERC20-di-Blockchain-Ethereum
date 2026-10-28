import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import random

from config import DEFAULT_DATASET


@st.cache_data
def load_data(uploaded_file=None):
    """Load dataset dari file CSV yang diupload atau file default"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, low_memory=False)
            st.success(f"Dataset berhasil diupload! Shape: {df.shape}")
        else:
            df = pd.read_csv(DEFAULT_DATASET, low_memory=False)
            st.info(f"Menggunakan dataset default: {DEFAULT_DATASET}")
        return df
    except FileNotFoundError:
        st.error(f"File tidak ditemukan. Silakan upload dataset.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


@st.cache_data
def preprocess_and_build_graph(df_raw, random_seed=42):
    """Preprocess data dan bangun graph"""
    df = df_raw.copy()
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    df['From_NameTag'].fillna('NA', inplace=True)
    df['To_NameTag'].fillna('NA', inplace=True)
    df['Method'].fillna('NA', inplace=True)
    
    df['Value (USD)'] = df['Value (USD)'].astype(str).str.replace(',', '', regex=False)
    df['Value (USD)'] = pd.to_numeric(df['Value (USD)'], errors='coerce').fillna(0)
    df['Amount'] = pd.to_numeric(df['Amount'].astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0)
    
    le_method = LabelEncoder()
    df['method_encoded'] = le_method.fit_transform(df['Method'].astype(str))
    
    df['is_from_phishing_tag'] = df['From_NameTag'].str.contains('Phishing', case=False, na=False).astype(int)
    df['is_to_phishing_tag'] = df['To_NameTag'].str.contains('Phishing', case=False, na=False).astype(int)
    df['is_method_claim_airdrop'] = df['Method'].str.contains('Airdrop|Claim', case=False, na=False).astype(int)
    df['is_zero_value_usd'] = (df['Value (USD)'] < 0.01).astype(int)
    
    if 'Txn Fee' not in df.columns:
        df['Txn Fee'] = 0
    else:
        df['Txn Fee'] = pd.to_numeric(df['Txn Fee'], errors='coerce').fillna(0)
    
    G_nx = nx.from_pandas_edgelist(df, 'From', 'To', create_using=nx.DiGraph())
    in_degree = dict(G_nx.in_degree())
    out_degree = dict(G_nx.out_degree())
    
    try:
        pagerank = nx.pagerank(G_nx, alpha=0.85, max_iter=100, tol=1.0e-4)
    except:
        pagerank = {node: 0 for node in G_nx.nodes()}
    
    all_addresses = pd.concat([df['From'], df['To']]).unique()
    address_to_id = {address: i for i, address in enumerate(all_addresses)}
    num_nodes = len(all_addresses)
    
    feature_cols = ['is_from_phishing_tag', 'is_to_phishing_tag', 'is_method_claim_airdrop', 
                    'is_zero_value_usd', 'method_encoded', 'Value (USD)', 'Amount', 'Txn Fee']
    
    node_features_list = []
    node_labels_list = []
    
    for i in range(num_nodes):
        address = all_addresses[i]
        node_in_degree = in_degree.get(address, 0)
        node_out_degree = out_degree.get(address, 0)
        node_pagerank = pagerank.get(address, 0)
        
        tx_as_from = df[df['From'] == address]
        tx_as_to = df[df['To'] == address]
        combined_tx = pd.concat([tx_as_from, tx_as_to])
        
        count_tx = len(combined_tx)
        related_labels = combined_tx['label'].tolist() if count_tx > 0 else []
        
        mean_features = np.zeros(len(feature_cols))
        sum_features = np.zeros(len(feature_cols))
        
        if count_tx > 0:
            try:
                numeric_cols = combined_tx[feature_cols].select_dtypes(include=np.number).columns
                if not combined_tx[numeric_cols].empty:
                    mean_features_calc = combined_tx[numeric_cols].mean(axis=0).fillna(0).values
                    sum_features_calc = combined_tx[numeric_cols].sum(axis=0).fillna(0).values
                    
                    if len(mean_features_calc) < len(feature_cols):
                        numeric_indices = [idx for idx, col in enumerate(feature_cols) if col in numeric_cols]
                        mean_features[numeric_indices] = mean_features_calc
                        sum_features[numeric_indices] = sum_features_calc
                    else:
                        mean_features = mean_features_calc
                        sum_features = sum_features_calc
            except:
                pass
        
        node_feature_vector = np.concatenate([
            [node_in_degree, node_out_degree, node_pagerank, count_tx],
            mean_features,
            sum_features
        ])
        
        node_features_list.append(node_feature_vector)
        
        if related_labels:
            node_label = Counter(related_labels).most_common(1)[0][0]
        else:
            node_label = 0
        
        node_labels_list.append(node_label)
    
    x_np = np.array(node_features_list)
    y_np = np.array(node_labels_list)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_np)
    
    source_nodes = df['From'].map(address_to_id).values
    target_nodes = df['To'].map(address_to_id).values
    valid_edge_mask = ~np.isnan(source_nodes) & ~np.isnan(target_nodes)
    source_nodes = source_nodes[valid_edge_mask].astype(int)
    target_nodes = target_nodes[valid_edge_mask].astype(int)
    edge_index_np = np.array([source_nodes, target_nodes])
    
    x = torch.tensor(x_scaled, dtype=torch.float)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    y = torch.tensor(y_np, dtype=torch.long)
    
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    
    num_nodes = graph_data.num_nodes
    indices = np.arange(num_nodes)
    
    try:
        train_idx, test_idx, y_train, _ = train_test_split(
            indices, y_np, test_size=0.3, random_state=random_seed, stratify=y_np
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.15, random_state=random_seed, stratify=y_train
        )
    except:
        train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=random_seed)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=random_seed)
    
    graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    graph_data.train_mask[train_idx] = True
    graph_data.val_mask[val_idx] = True
    graph_data.test_mask[test_idx] = True
    
    return df, graph_data, G_nx, all_addresses, address_to_id