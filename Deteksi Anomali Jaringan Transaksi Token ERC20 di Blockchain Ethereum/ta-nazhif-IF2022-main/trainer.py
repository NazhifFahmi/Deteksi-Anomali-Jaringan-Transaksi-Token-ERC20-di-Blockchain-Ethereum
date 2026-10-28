import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

from model import GraphSAGE
from config import LABEL_NAMES


def train_model(graph_data, config, seed=42):
    """Train GraphSAGE model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = graph_data.to(device)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=config['hidden_channels'],
        out_channels=len(LABEL_NAMES),
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    class_counts = torch.bincount(data.y)
    num_classes = len(LABEL_NAMES)
    
    if len(class_counts) < num_classes:
        padded_counts = torch.zeros(num_classes, dtype=torch.long)
        padded_counts[:len(class_counts)] = class_counts
        class_counts = padded_counts
    
    total_samples = class_counts.sum().float()
    class_weights = total_samples / (num_classes * class_counts.float() + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.nan_to_num(class_weights, nan=1.0)
    
    criterion = torch.nn.NLLLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    progress_bar = st.progress(0)
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    history = {
        'epoch': [], 'train_loss': [], 'train_f1': [],
        'val_loss': [], 'val_f1': []
    }
    
    best_val_f1 = -1
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        
        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, data.edge_index)
            
            pred_train = out_eval[data.train_mask].argmax(dim=1).cpu().numpy()
            y_true_train = data.y[data.train_mask].cpu().numpy()
            f1_train = f1_score(y_true_train, pred_train, average='macro', zero_division=0)
            
            pred_val = out_eval[data.val_mask].argmax(dim=1).cpu().numpy()
            y_true_val = data.y[data.val_mask].cpu().numpy()
            val_loss = F.nll_loss(out_eval[data.val_mask], data.y[data.val_mask], 
                                 weight=class_weights.to(device)).item()
            f1_val = f1_score(y_true_val, pred_val, average='macro', zero_division=0)
        
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_f1'].append(f1_train)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(f1_val)
        
        progress_bar.progress(epoch / config['epochs'])
        
        with metrics_placeholder.container():
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Epoch", f"{epoch}/{config['epochs']}")
            col2.metric("Train Loss", f"{train_loss:.4f}")
            col3.metric("Train F1", f"{f1_train:.4f}")
            col4.metric("Val Loss", f"{val_loss:.4f}")
            col5.metric("Val F1", f"{f1_val:.4f}")
        
        if epoch % 5 == 0 or epoch == 1 or epoch == config['epochs']:
            plot_data = pd.DataFrame(history)
            
            with chart_placeholder.container():
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                axes[0].plot(plot_data['epoch'], plot_data['train_loss'], 
                           label='Train', color='#3498db', linewidth=2)
                axes[0].plot(plot_data['epoch'], plot_data['val_loss'], 
                           label='Validation', color='#e74c3c', linewidth=2)
                axes[0].set_title('Training & Validation Loss', fontweight='bold')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(plot_data['epoch'], plot_data['train_f1'], 
                           label='Train', color='#2ecc71', linewidth=2)
                axes[1].plot(plot_data['epoch'], plot_data['val_f1'], 
                           label='Validation', color='#f39c12', linewidth=2)
                axes[1].set_title('Training & Validation F1 Score', fontweight='bold')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('F1 Score')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        
        if f1_val > best_val_f1:
            best_val_f1 = f1_val
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_f1, history


def run_multiple_evaluations(df, graph_data, G_nx, config, num_runs=5):
    """Jalankan evaluasi sebanyak num_runs kali"""
    all_results = []
    best_run_idx = -1
    best_f1_macro = -1
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for run_idx in range(num_runs):
        status_text.text(f"🚀 Menjalankan evaluasi {run_idx + 1}/{num_runs}...")
        
        seed = 42 + run_idx * 100
        
        model, best_val_f1, history = train_model(graph_data, config, seed=seed)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        data = graph_data.to(device)
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
        
        test_accuracy = accuracy_score(y_true, y_pred)
        test_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        test_f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        per_label_metrics = []
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
        for label in unique_labels:
            if label in LABEL_NAMES:
                label_name = LABEL_NAMES[label]
                
                precision = precision_score(y_true, y_pred, labels=[label], average='micro', zero_division=0)
                recall = recall_score(y_true, y_pred, labels=[label], average='micro', zero_division=0)
                f1 = f1_score(y_true, y_pred, labels=[label], average='micro', zero_division=0)
                
                per_label_metrics.append({
                    'Label': label_name,
                    'Precision': f"{precision:.4f}",
                    'Recall': f"{recall:.4f}",
                    'F1-Score': f"{f1:.4f}",
                    'Support': int(np.sum(y_true == label))
                })
        
        result = {
            'run': run_idx + 1,
            'seed': seed,
            'test_accuracy': test_accuracy,
            'test_f1_macro': test_f1_macro,
            'test_f1_weighted': test_f1_weighted,
            'per_label_metrics': per_label_metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'model': model,
            'best_val_f1': best_val_f1,
            'history': history
        }
        
        all_results.append(result)
        
        if test_f1_macro > best_f1_macro:
            best_f1_macro = test_f1_macro
            best_run_idx = run_idx
        
        progress_bar.progress((run_idx + 1) / num_runs)
    
    status_text.text("✅ Semua evaluasi selesai!")
    
    return all_results, best_run_idx