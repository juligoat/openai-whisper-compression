import datasets
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import Audio, display


def load_librispeech(num_samples=None):
    """Load LibriSpeech test-clean dataset.

    Args:
        num_samples (int, optional): Number of samples to load.
            If None, loads full dataset.

    Returns:
        datasets.Dataset: Dataset containing audio samples and transcriptions
    """
    if num_samples:
        # Stream partial dataset
        stream_dataset = datasets.load_dataset(
            "librispeech_asr", split="test.clean", streaming=True
        )
        dataset = datasets.Dataset.from_dict(
            {
                k: [sample[k] for sample in list(stream_dataset.take(num_samples))]
                for k in next(iter(stream_dataset)).keys()
            }
        )
    else:
        # Load full dataset
        dataset = datasets.load_dataset("librispeech_asr", "clean", split="test.clean")

    print(f"Loaded {len(dataset)} test samples")
    return dataset


def examine_dataset(dataset):
    """
    Examine LibriSpeech dataset characteristics.

    Args:
        dataset: LibriSpeech dataset from load_librispeech()
    """
    # Basic dataset info
    print("Dataset Overview:")
    print("-" * 50)
    print(f"Number of samples: {len(dataset)}")
    print(f"Features available: {list(dataset.features.keys())}")

    # Audio characteristics
    durations = [len(x["audio"]["array"]) / x["audio"]["sampling_rate"] for x in dataset]

    print("\nAudio Characteristics:")
    print("-" * 50)
    print(f"Sampling rate: {dataset[0]['audio']['sampling_rate']} Hz")
    print(f"Average duration: {np.mean(durations):.2f} seconds")
    print(f"Min duration: {min(durations):.2f} seconds")
    print(f"Max duration: {max(durations):.2f} seconds")

    # Text characteristics
    text_lengths = [len(x["text"].split()) for x in dataset]

    print("\nText Characteristics:")
    print("-" * 50)
    print(f"Average words per sample: {np.mean(text_lengths):.2f}")
    print(f"Min words: {min(text_lengths)}")
    print(f"Max words: {max(text_lengths)}")

    # Speaker information
    unique_speakers = len(set(x["speaker_id"] for x in dataset))
    print(f"\nNumber of unique speakers: {unique_speakers}")

    # Sample example
    print("\nExample Sample:")
    print("-" * 50)
    example = dataset[0]
    print(f"Text: {example['text']}")
    print(f"Speaker ID: {example['speaker_id']}")
    print(
        f"Duration: {len(example['audio']['array'])/example['audio']['sampling_rate']:.2f} seconds"
    )


def plot_dataset_distributions(dataset):
    """Plot distributions of dataset characteristics"""
    durations = [len(x["audio"]["array"]) / x["audio"]["sampling_rate"] for x in dataset]
    text_lengths = [len(x["text"].split()) for x in dataset]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Audio duration distribution
    sns.histplot(durations, bins=20, ax=axes[0])
    axes[0].set_title("Distribution of Audio Durations")
    axes[0].set_xlabel("Duration (seconds)")

    # Text length distribution
    sns.histplot(text_lengths, bins=20, ax=axes[1])
    axes[1].set_title("Distribution of Text Lengths")
    axes[1].set_xlabel("Number of Words")

    plt.tight_layout()
    plt.show()


def analyze_audio_characteristics(dataset, index=0):
    """
    Analyze audio characteristics of a specific sample from the dataset.

    Args:
        dataset: LibriSpeech dataset from load_librispeech().
        index: Index of the sample to analyze (default is 0).
    """
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Index {index} is out of bounds for the dataset (size: {len(dataset)})")

    # Get the specified sample
    sample = dataset[index]
    audio_array = np.array(sample["audio"]["array"])  # Convert to NumPy array
    sampling_rate = sample["audio"]["sampling_rate"]

    print(f"Analyzing Sample {index}")
    print("Audio Characteristics:")
    print("-" * 50)
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Audio shape: {audio_array.shape}")
    print(f"Duration: {len(audio_array)/sampling_rate:.2f} seconds")

    # Audio statistics
    print("\nAudio Statistics:")
    print(f"Mean amplitude: {np.mean(audio_array):.6f}")
    print(f"Max amplitude: {np.max(np.abs(audio_array)):.6f}")
    print(f"RMS energy: {np.sqrt(np.mean(audio_array**2)):.6f}")

    # Using librosa for additional analysis
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sampling_rate, n_mfcc=13)
    print(f"\nMFCC shape: {mfccs.shape}")

    # Visualize waveform and spectrogram
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"Waveform (Sample {index})")
    plt.plot(audio_array)

    plt.subplot(1, 2, 2)
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(audio_array)), ref=np.max),
        sr=sampling_rate,
        y_axis="log",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram (Sample {index})")

    plt.tight_layout()
    plt.show()


def play_sample(dataset, index=0):
    """
    Play audio sample from dataset and show its transcription.

    Args:
        dataset: LibriSpeech dataset
        index (int): Index of sample to play
    """
    sample = dataset[index]
    print("Text:", sample["text"])
    print(f"Duration: {len(sample['audio']['array'])/sample['audio']['sampling_rate']:.2f} seconds")
    print(f"Speaker ID: {sample['speaker_id']}")

    audio = Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
    display(audio)
