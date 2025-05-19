import librosa
import os
import numpy as np
import torch
from preprocess import sperate_magnitude_phase, combine_magnitdue_phase
from model import SVSRNN
import soundfile as sf


def separate_sources(song_filenames, output_directory = 'main-demo'):

    # Preprocess parameters
    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    num_rnn_layer = 4
    num_hidden_units = [256, 256, 256, 256]
    tensorboard_directory = 'graphs/svsrnn'
    clear_tensorboard = False
    model_directory = 'model'
    model_filename = 'svsrnn.pth'
    model_filepath = os.path.join(model_directory, model_filename)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    wavs_mono = list()
    for filename in song_filenames:
        wav_mono, _ = librosa.load(filename, sr = mir1k_sr, mono = True)
        wavs_mono.append(wav_mono)

    stfts_mono = list()
    for wav_mono in wavs_mono:
        stft_mono = librosa.stft(wav_mono, n_fft = n_fft, hop_length = hop_length)
        stfts_mono.append(stft_mono.transpose())

    model = SVSRNN(num_features = n_fft // 2 + 1, num_rnn_layer = num_rnn_layer, num_hidden_units = num_hidden_units, tensorboard_directory = tensorboard_directory, clear_tensorboard = clear_tensorboard)
    
    # Build the model with a sample input before loading weights
    sample_input = torch.zeros((1, stfts_mono[0].shape[0], n_fft // 2 + 1))
    _ = model(sample_input)  # This builds the model
    
    # Now load the weights
    model.load(filepath = model_filepath)

    for wav_filename, wav_mono, stft_mono in zip(song_filenames, wavs_mono, stfts_mono):

        wav_filename_dir = os.path.dirname(wav_filename)
        wav_filename_base = os.path.basename(wav_filename)
        wav_mono_filename = wav_filename_base.split('.')[0] + '_mono.wav'
        wav_src1_hat_filename = wav_filename_base.split('.')[0] + '_background.wav'
        wav_src2_hat_filename = wav_filename_base.split('.')[0] + '_vocals.wav'
        wav_mono_filepath = os.path.join(output_directory, wav_mono_filename)
        wav_src1_hat_filepath = os.path.join(output_directory, wav_src1_hat_filename)
        wav_src2_hat_filepath = os.path.join(output_directory, wav_src2_hat_filename)

        print('Processing %s ...' % wav_filename_base)

        stft_mono_magnitude, stft_mono_phase = sperate_magnitude_phase(data = stft_mono)
        stft_mono_magnitude = torch.tensor(np.array([stft_mono_magnitude]), dtype=torch.float32)

        y1_pred, y2_pred = model.test(x = stft_mono_magnitude)

        # Convert predictions back to numpy for processing
        y1_pred = y1_pred.detach().numpy()
        y2_pred = y2_pred.detach().numpy()

        # ISTFT with the phase from mono
        y1_stft_hat = combine_magnitdue_phase(magnitudes = y1_pred[0], phases = stft_mono_phase)
        y2_stft_hat = combine_magnitdue_phase(magnitudes = y2_pred[0], phases = stft_mono_phase)

        y1_stft_hat = y1_stft_hat.T
        y2_stft_hat = y2_stft_hat.T

        y1_hat = librosa.istft(y1_stft_hat, hop_length=hop_length)
        y2_hat = librosa.istft(y2_stft_hat, hop_length=hop_length)

        sf.write(wav_mono_filepath, wav_mono, mir1k_sr)
        sf.write(wav_src1_hat_filepath, y1_hat, mir1k_sr)
        sf.write(wav_src2_hat_filepath, y2_hat, mir1k_sr)



if __name__ == '__main__':

    songs_dir = 'songs'
    song_filenames = list()
    for file in os.listdir(songs_dir):
        if file.endswith('.mp3'):
            song_filenames.append(os.path.join(songs_dir, file))

    separate_sources(song_filenames = song_filenames, output_directory = 'main-demo')