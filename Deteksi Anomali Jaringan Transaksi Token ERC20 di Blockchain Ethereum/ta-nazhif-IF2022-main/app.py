import streamlit as st
import torch
import warnings
warnings.filterwarnings('ignore')

from config import DEFAULT_CONFIG, DEFAULT_DATASET
from data_loader import load_data, preprocess_and_build_graph
from trainer import run_multiple_evaluations
from visualizations import (
    show_data_overview, show_label_distribution, show_graph_statistics,
    show_data_split, show_graph_visualization, show_evaluation_tables_per_label,
    show_classification_report, show_confusion_matrix, show_tsne_all_nodes,
    show_detection_results_comparison
)


def main():
    st.set_page_config(page_title="GNN Anomaly Detection", layout="wide")
    
    st.title("🔬 Sistem Deteksi Anomali Blockchain dengan GraphSAGE")
    
    # ==================== SIDEBAR: DATASET ====================
    st.sidebar.header("📁 Input Dataset")
    
    dataset_option = st.sidebar.radio(
        "Pilihan Dataset:",
        ["Gunakan Dataset Default", "Upload Dataset Baru"]
    )
    
    df = None
    
    if dataset_option == "Upload Dataset Baru":
        uploaded_file = st.sidebar.file_uploader(
            "Upload file CSV dataset", 
            type=['csv'],
            help="Upload dataset transaksi blockchain dalam format CSV"
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
        else:
            st.info("Silakan upload dataset atau gunakan dataset default")
            st.stop()
    else:
        df = load_data()
    
    if df is None:
        st.error("Gagal memuat dataset. Silakan coba lagi.")
        st.stop()
    
    # ==================== SIDEBAR: MODEL PARAMETERS ====================
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Konfigurasi Model")
    
    with st.sidebar.expander("🔧 Parameter Model", expanded=False):
        hidden_channels = st.number_input(
            "Hidden Channels",
            min_value=32,
            max_value=512,
            value=DEFAULT_CONFIG['hidden_channels'],
            step=32,
            help="Jumlah hidden channels dalam layer GraphSAGE"
        )
        
        num_layers = st.number_input(
            "Number of Layers",
            min_value=2,
            max_value=10,
            value=DEFAULT_CONFIG['num_layers'],
            step=1,
            help="Jumlah layer GraphSAGE"
        )
        
        dropout = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=0.9,
            value=DEFAULT_CONFIG['dropout'],
            step=0.1,
            help="Dropout rate untuk regularisasi"
        )
    
    with st.sidebar.expander("📚 Parameter Training", expanded=False):
        epochs = st.number_input(
            "Epochs",
            min_value=10,
            max_value=1000,
            value=DEFAULT_CONFIG['epochs'],
            step=10,
            help="Jumlah epoch training"
        )
        
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=DEFAULT_CONFIG['learning_rate'],
            step=0.0005,
            format="%.4f",
            help="Learning rate untuk optimizer"
        )
        
        weight_decay = st.number_input(
            "Weight Decay",
            min_value=0.0,
            max_value=0.01,
            value=DEFAULT_CONFIG['weight_decay'],
            step=0.0001,
            format="%.4f",
            help="Weight decay (L2 regularization)"
        )
        
        patience = st.number_input(
            "Early Stopping Patience",
            min_value=5,
            max_value=50,
            value=DEFAULT_CONFIG['patience'],
            step=5,
            help="Jumlah epoch tanpa improvement sebelum early stopping"
        )
    
    with st.sidebar.expander("🔄 Parameter Evaluasi", expanded=False):
        num_runs = st.number_input(
            "Number of Runs",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Jumlah evaluasi yang akan dijalankan"
        )
    
    config = {
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'dropout': dropout,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'patience': patience
    }
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Konfigurasi Saat Ini")
    st.sidebar.json(config)
    
    if st.sidebar.button("🔄 Reset ke Default", use_container_width=True):
        st.rerun()
    
    # ==================== MAIN CONTENT ====================
    st.markdown(f"**Dataset yang digunakan:** {'Uploaded File' if dataset_option == 'Upload Dataset Baru' else DEFAULT_DATASET}")
    st.markdown("---")
    
    show_data_overview(df)
    
    st.markdown("---")
    show_label_distribution(df)
    
    st.markdown("---")
    
    if st.button("🚀 Mulai Proses Training & Evaluasi", type="primary", use_container_width=True):
        with st.spinner("Memproses data dan membangun graph..."):
            df_processed, graph_data, G_nx, all_addresses, address_to_id = preprocess_and_build_graph(df)
        
        st.success("✅ Data berhasil diproses!")
        
        st.markdown("---")
        show_graph_statistics(graph_data)
        
        st.markdown("---")
        show_data_split(graph_data)
        
        st.markdown("---")
        show_graph_visualization(graph_data, G_nx)
        
        st.markdown("---")
        st.subheader("🎯 Evaluasi Model")
        
        all_results, best_run_idx = run_multiple_evaluations(
            df_processed, graph_data, G_nx, config, num_runs=num_runs
        )
        
        show_evaluation_tables_per_label(all_results)
        
        st.markdown("---")
        st.subheader(f"🏆 Hasil Terbaik (Run {best_run_idx + 1})")
        
        best_result = all_results[best_run_idx]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Run", f"{best_run_idx + 1}")
        with col2:
            st.metric("Test Accuracy", f"{best_result['test_accuracy']:.4f}")
        with col3:
            st.metric("Test F1 Macro", f"{best_result['test_f1_macro']:.4f}")
        with col4:
            st.metric("Test F1 Weighted", f"{best_result['test_f1_weighted']:.4f}")
        
        st.markdown("---")
        st.subheader(f"📊 Visualisasi Hasil Terbaik (Run {best_run_idx + 1})")
        
        show_classification_report(best_result['y_true'], best_result['y_pred'])
        
        st.markdown("---")
        show_confusion_matrix(best_result['y_true'], best_result['y_pred'])
        
        st.markdown("---")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        show_tsne_all_nodes(best_result['model'], graph_data, device)
        
        st.markdown("---")
        detection_df = show_detection_results_comparison(
            df_processed, graph_data, best_result['model'], 
            all_addresses, address_to_id, device
        )
        
        st.markdown("---")
        st.success(f"✅ Semua evaluasi selesai! Hasil terbaik: Run {best_run_idx + 1} dengan F1 Macro: {best_result['test_f1_macro']:.4f}")


if __name__ == "__main__":
    main()