# This file is kept as a placeholder for future utility functions if needed.

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

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

def plot_spectrogram(audio_path, output_path, sr=16000):
    """Plot the spectrogram of an audio file."""
    y, sr = librosa.load(audio_path, sr=sr)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()

def plot_separation_comparison(mixed_audio, separated_vocals, separated_background, output_path, sr=16000):
    """Plot waveform comparison of mixed audio and separated sources."""
    plt.figure(figsize=(15, 10))
    
    # Plot mixed audio
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(mixed_audio, sr=sr)
    plt.title('Mixed Audio')
    
    # Plot separated vocals
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(separated_vocals, sr=sr)
    plt.title('Separated Vocals')
    
    # Plot separated background
    plt.subplot(3, 1, 3)
    librosa.display.waveshow(separated_background, sr=sr)
    plt.title('Separated Background')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()

def plot_si_snr_comparison(original_sources, separated_sources, output_path):
    """Plot SI-SNR comparison between original and separated sources."""
    plt.figure(figsize=(10, 6))
    
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, original_sources, width, label='Original Sources')
    plt.bar(x + width/2, separated_sources, width, label='Separated Sources')
    
    plt.xlabel('Source Type')
    plt.ylabel('SI-SNR (dB)')
    plt.title('SI-SNR Comparison')
    plt.xticks(x, ['Vocals', 'Background'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()

if __name__ == '__main__':
    # Example usage
    log_path = os.path.join('log', 'train_log.csv')
    output_path = os.path.join('figures', 'train_validation_loss.png')
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plot_train_and_validation_loss(log_path, output_path)

    # For spectrogram
    plot_spectrogram('songs\sample-original.mp3', 'figures/spectrogram.png')

    # For separation comparison
    mixed_audio = librosa.load('main-demo\sample-original_mono.wav', sr=16000)[0]
    vocals = librosa.load('main-demo\sample-original_vocals.wav', sr=16000)[0]
    background = librosa.load('main-demo\sample-original_background.wav', sr=16000)[0]
    plot_separation_comparison(mixed_audio, vocals, background, 'figures/separation_comparison.png')

    # For SI-SNR comparison
    original_sources = [20.5, 18.3]  # Example SI-SNR values for original vocals and background
    separated_sources = [15.2, 12.8]  # Example SI-SNR values for separated sources
    plot_si_snr_comparison(original_sources, separated_sources, 'figures/si_snr_comparison.png')
