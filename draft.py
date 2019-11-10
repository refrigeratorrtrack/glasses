# Drafts for different data-processing scripts
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

def main():
    command = input("Enter command or 'exit': ")

    while command != "exit":
        if command == "contrast":
            path = input("Enter path to file: ")
            spectral_contrast(path)
        elif command == "rolloff":
            path = input("Enter path to file: ")
            spectral_rolloff(path)

        command = input("Enter path or 'exit': ")


def spectral_contrast(path_to_file):
    y, sr = librosa.load(path_to_file)
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power spectrogram')
    plt.subplot(2, 1, 2)
    librosa.display.specshow(contrast, x_axis='time')
    plt.colorbar()
    plt.ylabel('Frequency bands')
    plt.title('Spectral contrast')
    plt.tight_layout()
    plt.show()


def spectral_rolloff(path_to_file):
    y, sr = librosa.load(path_to_file)
    S, phase = librosa.magphase(librosa.stft(y))
    # Approximate maximum frequencies with roll_percent=0.85 (default)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # Approximate minimum frequencies with roll_percent=0.1
    # rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.1)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogy(rolloff.T, label='Roll-off frequency')
    plt.ylabel('Hz')
    plt.xticks([])
    plt.xlim([0, rolloff.shape[-1]])
    plt.legend()
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
    plt.title('log Power spectrogram')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
