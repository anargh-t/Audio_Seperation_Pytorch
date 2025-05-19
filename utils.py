import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_train_and_validation_loss(log_path, output_path):
    df = pd.read_csv(log_path)
    plt.figure(figsize=(12, 8))
    plt.plot(df['iteration'], df['train_loss'], label='Training Loss', color='red')
    plt.plot(df['iteration'], df['validation_loss'], label='Validation Loss', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (SI-SNR)')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()

if __name__ == '__main__':
    log_path = os.path.join('log', 'train_log.csv')
    output_path = os.path.join('figures', 'train_validation_loss.png')
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plot_train_and_validation_loss(log_path, output_path)
