import os
import time
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


def main():
    data = dict.fromkeys(os.listdir(path="./data"), [])

    for key in data.keys():
        data[key] = os.listdir(path="./data/" + key)

    for key in data.keys():
        start_time = time.time()

        for file in data[key]:
            save_directory = "./data/" + key + "/" + file.split(".")[0]
            os.mkdir(save_directory)
            x, sr = librosa.load("./data/" + key + "/" + file)

            amplitude_time(x, sr, save_directory)
            spectrogram(x, sr, save_directory)
            amplitude_frequency(x, sr, save_directory)

        print("./data/" + key + " processed successfully!")
        print("It took " + str(time.time() - start_time) + " second.")


def amplitude_time(x, sr, file_path):
    """ Draw amplitude-time. """

    amplitudeTime = plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    amplitudeTime.savefig(file_path + "/amplitudeTime.png", dpi=1000)
    plt.close(amplitudeTime)


def spectrogram(x, sr, file_path):
    """ Draw spectrogram. """

    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    spectrogram = plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.title("Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    spectrogram.savefig(file_path + "/spectrogram.png", dpi=600)
    plt.close(spectrogram)


def amplitude_frequency(x, sr, file_path):
    """ Draw amplitude-frequency. """

    n_fft = 2048
    S = librosa.stft(x, n_fft=n_fft, hop_length=n_fft//2)
    D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    D_AVG = np.mean(D, axis=1)
    amplitudeFrequency = plt.figure(figsize=(14, 5))
    plt.bar(np.arange(D_AVG.shape[0]), D_AVG)
    x_ticks_positions = [n for n in range(0, n_fft // 2, n_fft // 16)]
    x_ticks_labels = [str(sr / 2048 * n) + 'Hz' for n in x_ticks_positions]
    plt.xticks(x_ticks_positions, x_ticks_labels)
    plt.xlabel('Frequency')
    plt.ylabel('dB')
    amplitudeFrequency.savefig(file_path + "/amplitudeFrequency.png", dpi=600)
    plt.close(amplitudeFrequency)


if __name__ == '__main__':
    main()
