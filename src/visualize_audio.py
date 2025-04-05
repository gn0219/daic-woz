import os
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
plt.style.use("ggplot")

def seconds_to_minsec(x, pos):
    minutes = int(x) // 60
    seconds = int(x) % 60
    return f"{minutes}:{seconds:02d}"

def load_audio(uid: str, wav_dir='wav_files', sr_target=16000, max_duration=None):
    wav_path = Path(wav_dir) / f"{uid}_AUDIO.wav"
    if max_duration is not None:
        y, sr = librosa.load(wav_path, sr=sr_target, duration=max_duration)
    else:
        y, sr = librosa.load(wav_path, sr=sr_target)
    return y, sr

def plot_audio(uid: str, gender: str, label: str):
    y, sr = load_audio(uid)
    t = np.linspace(0, len(y) / sr, num=len(y))

    n_fft = 1024
    hop_length = 512
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1], wspace=0.05, hspace=0.4)

    # ▶ 1행: waveform 
    ax0 = fig.add_subplot(gs[0, 0])  # row 0, col 0
    ax0.plot(t, y, color="tomato", linewidth=0.8)
    ax0.set_title("Waveform")
    ax0.set_ylabel("Amplitude")
    ax0.set_xlim([0, t[-1]])
    ax0.xaxis.set_major_formatter(FuncFormatter(seconds_to_minsec))

    # ▶ 2행: spectrogram 
    ax1 = fig.add_subplot(gs[1, 0])  # row 1, col 0
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                                   x_axis='time', y_axis='hz', cmap='viridis', ax=ax1)
    ax1.set_title("Log Spectrogram")
    ax1.set_xlabel("Time (min:sec)")
    ax1.set_ylabel("Frequency (Hz)")

    # ▶ Colorbar 따로 오른쪽 col=1에 위치
    cax = fig.add_subplot(gs[1, 1])  # row 1, col 1
    cbar = fig.colorbar(img, cax=cax, format="%+2.0f dB")
    cbar.set_label("dB")

    if gender is not None and label is not None:
        gender = "Female" if gender == 0 else "Male"
        label = "ND" if label == 0 else "D"
        fig.suptitle(f"PID: {uid} - {gender}, {label}", fontsize=15)
    else:
        fig.suptitle(f"PID: {uid} - Audio Visualization", fontsize=15)
    plt.tight_layout()
    plt.show()