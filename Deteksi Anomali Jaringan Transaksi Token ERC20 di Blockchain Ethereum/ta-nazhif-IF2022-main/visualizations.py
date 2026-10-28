import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import torch_geometric.utils

from config import LABEL_NAMES


def show_data_overview(df):
    """Visualisasi: Overview Dataset"""
    st.subheader("Overview Dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transaksi", f"{df.shape[0]:,}")
    with col2:
        st.metric("Jumlah Kolom", f"{df.shape[1]:,}")
    with col3:
        memory = df.memory_usage(deep=True).sum()/1024**2
        st.metric("Ukuran Data", f"{memory:.2f} MB")
    
    st.dataframe(df, use_container_width=True, height=400)


def show_label_distribution(df):
    """Visualisasi: Distribusi Label Kelas"""
    st.subheader("Distribusi Label Kelas")
    
    label_counts = df['label'].value_counts().sort_index()
    label_names = [LABEL_NAMES.get(x, f"Label {x}") for x in label_counts.index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(label_names, label_counts.values, color=sns.color_palette("viridis", len(label_names)), 
                  edgecolor='black', linewidth=1.5)
    ax.set_title("Distribusi Kelas Transaksi", fontsize=14, fontweight='bold')
    ax.set_ylabel("Jumlah Transaksi", fontsize=12)
    ax.set_xlabel("Label Kelas", fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, 
               f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_graph_statistics(graph_data):
    """Visualisasi: Statistik Graph"""
    st.subheader("Statistik Graph")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jumlah Node", f"{graph_data.num_nodes:,}")
    with col2:
        st.metric("Jumlah Edge", f"{graph_data.num_edges:,}")
    with col3:
        st.metric("Fitur per Node", graph_data.num_node_features)
    with col4:
        avg_degree = graph_data.num_edges / graph_data.num_nodes if graph_data.num_nodes > 0 else 0
        st.metric("Rata-rata Degree", f"{avg_degree:.2f}")


def show_data_split(graph_data):
    """Visualisasi: Pembagian Data (Train/Val/Test)"""
    st.subheader("Pembagian Data (Train/Val/Test)")
    
    splits = ['Train', 'Validation', 'Test']
    counts = [
        graph_data.train_mask.sum().item(),
        graph_data.val_mask.sum().item(),
        graph_data.test_mask.sum().item()
    ]
    total = graph_data.num_nodes
    percentages = [(c/total*100) if total > 0 else 0 for c in counts]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(splits, counts, color=['#3498db', '#e67e22', '#2ecc71'], 
                  edgecolor='black', linewidth=1.5)
    ax.set_title('Distribusi Data Split', fontsize=14, fontweight='bold')
    ax.set_ylabel('Jumlah Node', fontsize=12)
    
    for bar, count, perc in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
               f'{count:,}\n({perc:.1f}%)', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_confusion_matrix(y_true, y_pred):
    """Visualisasi: Confusion Matrix"""
    st.subheader("🎯 Confusion Matrix")
    
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = [LABEL_NAMES.get(i, f'Label {i}') for i in unique_labels]
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=target_names, yticklabels=target_names,
               ax=ax, cbar_kws={'label': 'Count'}, annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_title("Confusion Matrix (Test Set)", fontweight='bold', fontsize=14)
    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_classification_report(y_true, y_pred):
    """Visualisasi: Classification Report"""
    st.subheader("Classification Report")
    
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = [LABEL_NAMES.get(i, f'Label {i}') for i in unique_labels]
    
    report_dict = classification_report(y_true, y_pred, target_names=target_names, 
                                       labels=unique_labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df[['precision', 'recall', 'f1-score']] = \
        report_df[['precision', 'recall', 'f1-score']].applymap('{:.4f}'.format)
    report_df['support'] = report_df['support'].astype(int)
    
    st.dataframe(report_df, use_container_width=True)


def show_tsne_all_nodes(model, graph_data, device):
    """Visualisasi: t-SNE Projection (Semua Node)"""
    st.subheader("t-SNE Projection")
    
    with st.spinner("Menghitung embedding untuk semua node..."):
        model.eval()
        data = graph_data.to(device)
        
        with torch.no_grad():
            embeddings_all = model.get_embedding(data.x, data.edge_index).cpu().numpy()
            labels_all = data.y.cpu().numpy()
        
        perplexity = min(30.0, max(5.0, float(len(embeddings_all) - 1)))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                   max_iter=1000, learning_rate=200.0)
        embeddings_2d = tsne.fit_transform(embeddings_all)
        
        color_palette = sns.color_palette("Set1", n_colors=4)
        label_to_color = {
            0: color_palette[0],
            1: color_palette[1],
            2: color_palette[2],
            3: color_palette[3]
        }
        
        fig, ax = plt.subplots(figsize=(16, 12))
        unique_labels = sorted(np.unique(labels_all))
        
        for label in unique_labels:
            mask = labels_all == label
            color = label_to_color.get(label, 'grey')
            label_name = LABEL_NAMES.get(label, f'Label {label}')
            size = 60 if label == 0 else 100
            
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[color], label=label_name,
                      alpha=0.7, s=size, edgecolors='black', linewidths=0.8)
        
        ax.set_title(f't-SNE Projection ({len(embeddings_all):,} nodes)',
                   fontweight='bold', fontsize=16, pad=20)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=13, fontweight='bold')
        ax.set_ylabel("t-SNE Dimension 2", fontsize=13, fontweight='bold')
        
        legend = ax.legend(title="Kelas Node", fontsize=12, title_fontsize=13, 
                          loc='best', frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def show_graph_visualization(graph_data, G_nx):
    """Visualisasi: Struktur Graph Network"""
    st.subheader("🕸️ Struktur Graph Network")
    
    num_nodes_total = graph_data.num_nodes
    num_edges_total = graph_data.num_edges
    
    if num_edges_total == 0 or num_nodes_total == 0:
        st.warning("Graph kosong, tidak dapat divisualisasikan.")
        return
    
    with st.spinner("Membuat visualisasi graph..."):
        try:
            G_to_draw = torch_geometric.utils.to_networkx(graph_data, to_undirected=False)
        except Exception as e:
            st.error(f"Gagal mengonversi ke NetworkX: {e}")
            return
        
        labels_np = graph_data.y.numpy()
        color_palette = sns.color_palette("Set1", n_colors=4)
        label_to_color = {0: color_palette[0], 1: color_palette[1],
                         2: color_palette[2], 3: color_palette[3]}
        default_color = 'grey'
        
        color_map = [label_to_color.get(label, default_color) for label in labels_np]
        node_sizes = [40 if label == 0 else 80 for label in labels_np]
        edge_color = '#666666'
        
        fig, ax = plt.subplots(figsize=(18, 16))
        
        st.info(f"📈 Menampilkan {num_nodes_total:,} node dan {num_edges_total:,} edge...")
        pos = nx.spring_layout(G_to_draw, k=1.0 / np.sqrt(num_nodes_total), iterations=100, seed=42)
        
        nx.draw_networkx_edges(
            G_to_draw, pos, ax=ax, alpha=0.4,
            edge_color=edge_color, width=0.8, arrows=False
        )
        
        nx.draw_networkx_nodes(
            G_to_draw, pos, ax=ax,
            node_color=color_map, node_size=node_sizes,
            edgecolors='black', linewidths=0.5
        )
        
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color=label_to_color.get(label, default_color),
                          label=LABEL_NAMES.get(label, f'Label {label}'))
            for label in sorted(np.unique(labels_np))
        ]
        ax.legend(handles=legend_handles, loc='upper right',
                 title="Kelas Node", fontsize=11, title_fontsize=12)
        
        ax.set_title("Visualisasi Graph Network Lengkap", fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def show_detection_results_comparison(df, graph_data, model, all_addresses, address_to_id, device):
    """Tampilkan tabel hasil deteksi antara actual class dan predicted class"""
    st.subheader("🔍 Hasil Deteksi: Actual Class vs Predicted Class")
    
    with st.spinner("Menghasilkan prediksi untuk semua node..."):
        model.eval()
        data = graph_data.to(device)
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            predictions = out.argmax(dim=1).cpu().numpy()
            probabilities = torch.exp(out).cpu().numpy()
        
        detection_df = pd.DataFrame({
            'Node_ID': range(len(all_addresses)),
            'Address': all_addresses,
            'Actual_Class': [LABEL_NAMES.get(l, f'Unknown {l}') for l in graph_data.y.cpu().numpy()],
            'Predicted_Class': [LABEL_NAMES.get(l, f'Unknown {l}') for l in predictions],
            'Match': graph_data.y.cpu().numpy() == predictions,
            'Max_Confidence': probabilities.max(axis=1),
        })
        
        display_df = detection_df.copy()
        display_df['Match'] = display_df['Match'].map({True: '✅', False: '❌'})
        display_df['Max_Confidence'] = display_df['Max_Confidence'].apply(lambda x: f"{x:.4f}")
        display_df['Address'] = display_df['Address'].apply(lambda x: f"{x[:10]}...{x[-8:]}" if len(str(x)) > 20 else x)
        
        st.dataframe(display_df, use_container_width=True, height=600)
        
    return detection_df


def show_evaluation_tables_per_label(all_results):
    """Tampilkan tabel evaluasi per label untuk setiap run"""
    st.subheader("Hasil Evaluasi Per Label")
    
    results_df = pd.DataFrame(all_results)
    
    st.write("### Ringkasan Performa")
    summary_df = results_df.groupby('run').agg({
        'test_accuracy': 'mean',
        'test_f1_macro': 'mean',
        'test_f1_weighted': 'mean'
    }).reset_index()
    summary_df.columns = ['Run', 'Accuracy', 'F1 Macro', 'F1 Weighted']
    summary_df = summary_df.round(4)
    
    st.dataframe(summary_df, use_container_width=True)
    
    for run_idx in range(len(all_results)):
        with st.expander(f"Detail Run {run_idx + 1}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"Run {run_idx + 1} - Accuracy", 
                         f"{all_results[run_idx]['test_accuracy']:.4f}")
            with col2:
                st.metric(f"Run {run_idx + 1} - F1 Macro", 
                         f"{all_results[run_idx]['test_f1_macro']:.4f}")
            with col3:
                st.metric(f"Run {run_idx + 1} - F1 Weighted", 
                         f"{all_results[run_idx]['test_f1_weighted']:.4f}")
            
            per_label_df = pd.DataFrame(all_results[run_idx]['per_label_metrics'])
            st.dataframe(per_label_df, use_container_width=True)