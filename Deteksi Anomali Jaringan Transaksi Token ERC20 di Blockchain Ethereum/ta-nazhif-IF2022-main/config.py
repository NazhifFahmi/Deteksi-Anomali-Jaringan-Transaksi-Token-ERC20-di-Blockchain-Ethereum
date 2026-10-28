# Dataset default
DEFAULT_DATASET = "transaksiERC20.csv"

# Label names
LABEL_NAMES = {
    0: 'Normal', 
    1: 'Ice Phishing', 
    2: 'NFT Scam', 
    3: 'Reward Scam'
}

# Default model configuration
DEFAULT_CONFIG = {
    'hidden_channels': 128,
    'num_layers': 3,
    'dropout': 0.5,
    'epochs': 200,
    'learning_rate': 0.005,
    'weight_decay': 5e-4,
    'patience': 15
}