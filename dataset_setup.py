import os
import pandas as pd
import gdown
import zipfile
from glob import glob
import torchaudio
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Audio, Features, ClassLabel, Value
from sklearn.model_selection import train_test_split
import gc
import librosa
import numpy as np


def create_and_push_hf_dataset(
    old_data_file_id="1UHsGP0tBmuObxE7D6qmcH5-lIIpz0BMC",
    new_data_file_id="1vo0TrZ2OFNHzCIVeQT4kanc26HHFnDPe",
    dataset_name="egyptian-dialect-audio",
    min_duration=2.0,
    max_duration=30.0,
    test_size=0.3,
    val_size=0.5,
    random_state=42,
    download_path="./",
    cleanup_zip=True,
    push_to_hub=True,
    private=False,
    skip_download=False
):
    """
    Create Egyptian dialect audio dataset and push it to Hugging Face Hub.
    
    Args:
        old_data_file_id (str): Google Drive file ID for the old dataset
        new_data_file_id (str): Google Drive file ID for the new dataset
        dataset_name (str): Name for the dataset on Hugging Face Hub
        min_duration (float): Minimum audio duration in seconds
        max_duration (float): Maximum audio duration in seconds
        test_size (float): Proportion for testing
        val_size (float): Proportion of test set for validation
        random_state (int): Random seed for reproducible splits
        download_path (str): Path to download and extract files
        cleanup_zip (bool): Whether to remove zip files after extraction
        push_to_hub (bool): Whether to push to Hugging Face Hub
        private (bool): Whether to make the dataset private
        skip_download (bool): Skip download if files already exist
    
    Returns:
        DatasetDict: The created dataset with train/validation/test splits
    """
    
    print("Starting dataset creation and Hub upload...")
    
    # Change to download directory
    original_dir = os.getcwd()
    os.chdir(download_path)
    
    try:
        # Check if audio files already exist
        existing_audio = glob("*/*.wav") + glob("*/*/*.wav") + glob("*/*_cut.wav")
        
        if len(existing_audio) > 0 and skip_download:
            print(f"Found {len(existing_audio)} existing audio files, skipping download...")
        else:
            # Download and extract datasets only if needed
            if not skip_download:
                print("Downloading datasets...")
                _download_and_extract_data(old_data_file_id, new_data_file_id, cleanup_zip)
            else:
                print("Skipping download as requested...")
        
        # Create DataFrame with audio files
        print("Creating DataFrame...")
        data_df = _create_dataframe()
        
        # Calculate audio durations and show initial statistics
        print("Calculating audio durations...")
        data_df = _calculate_audio_durations(data_df)
        
        # Show comprehensive statistics before filtering
        _analyze_dataset_statistics(data_df, min_duration=min_duration, max_duration=max_duration)
        
        # Filter by duration
        print("Filtering by audio duration...")
        data_df = _filter_by_duration(data_df, min_duration, max_duration)
        
        # Split dataset
        print("Splitting dataset...")
        train_df, val_df, test_df = _split_dataset(data_df, test_size, val_size, random_state)
        
        # Show statistics after splitting
        _analyze_dataset_statistics(data_df, train_df, val_df, test_df, min_duration, max_duration)
        
        # Create label mappings
        unique_labels = sorted(data_df['label'].unique())
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        print(f"Label mappings: {label2id}")
        
        # Convert to Hugging Face Datasets with proper features
        print("Converting to Hugging Face Datasets...")
        dataset_dict = _create_hf_datasets(train_df, val_df, test_df, unique_labels)
        
        # Add dataset metadata
        dataset_dict = _add_dataset_metadata(dataset_dict, label2id, id2label)
        
        if push_to_hub:
            print(f"Pushing dataset to Hugging Face Hub as '{dataset_name}'...")
            dataset_dict.push_to_hub(dataset_name, private=private)
            print(f"Dataset successfully pushed to: https://huggingface.co/datasets/{dataset_name}")
        
        # Clean up memory
        gc.collect()
        
        print("Dataset creation completed successfully!")
        _print_dataset_info(dataset_dict)
        
        return dataset_dict
        
    finally:
        # Return to original directory
        os.chdir(original_dir)
        
        
def _create_hf_datasets(train_df, val_df, test_df, unique_labels):
    """Create Hugging Face datasets with proper features."""
    
    # Define features schema
    features = Features({
        'audio': Audio(sampling_rate=16000),
        'label': ClassLabel(names=unique_labels),
        'audio_path': Value('string'),
    })
    
    # Load audio function
    def load_audio_data(example):
        audio_path = example['audio_path']
        try:
            # Load audio at 16kHz sampling rate
            audio, sr = librosa.load(audio_path, sr=16000)
            return {
                'audio': {'array': audio, 'sampling_rate': 16000},
                'label': example['label'],
                'audio_path': audio_path
            }
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return empty audio array if loading fails
            return {
                'audio': {'array': np.array([]), 'sampling_rate': 16000},
                'label': example['label'],
                'audio_path': audio_path
            }
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df[['audio_path', 'label']].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[['audio_path', 'label']].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[['audio_path', 'label']].reset_index(drop=True))
    
    # Load audio data
    print("Loading audio data for training set...")
    train_dataset = train_dataset.map(load_audio_data, num_proc=1)
    
    print("Loading audio data for validation set...")
    val_dataset = val_dataset.map(load_audio_data, num_proc=1)
    
    print("Loading audio data for test set...")
    test_dataset = test_dataset.map(load_audio_data, num_proc=1)
    
    # Remove the problematic cast operation for now and let datasets handle the schema
    # train_dataset = train_dataset.cast(features)
    # val_dataset = val_dataset.cast(features)
    # test_dataset = test_dataset.cast(features)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    return dataset_dict

def _analyze_dataset_statistics(data_df, train_df=None, val_df=None, test_df=None, min_duration=2.0, max_duration=30.0):
    """
    Analyze and display comprehensive statistics about the dataset.
    
    Args:
        data_df: Main dataframe with all samples
        train_df, val_df, test_df: Split dataframes (optional)
        min_duration, max_duration: Duration thresholds for analysis
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE DATASET STATISTICS")
    print("="*70)
    
    # Basic dataset info
    print(f"\n📊 BASIC DATASET INFO:")
    print(f"   Total samples: {len(data_df)}")
    print(f"   Total classes: {data_df['label'].nunique()}")
    print(f"   Classes: {sorted(data_df['label'].unique())}")
    
    # Class distribution
    print(f"\n📈 CLASS DISTRIBUTION:")
    class_counts = data_df['label'].value_counts().sort_index()
    for label, count in class_counts.items():
        percentage = (count / len(data_df)) * 100
        print(f"   {label:10}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Duration statistics
    if 'audio_length' in data_df.columns:
        print(f"\n⏱️  DURATION STATISTICS:")
        duration_stats = data_df['audio_length'].describe()
        print(f"   Mean duration: {duration_stats['mean']:6.2f}s")
        print(f"   Std duration:  {duration_stats['std']:6.2f}s")
        print(f"   Min duration:  {duration_stats['min']:6.2f}s")
        print(f"   Max duration:  {duration_stats['max']:6.2f}s")
        print(f"   Median:        {duration_stats['50%']:6.2f}s")
        print(f"   25th percentile: {duration_stats['25%']:6.2f}s")
        print(f"   75th percentile: {duration_stats['75%']:6.2f}s")
        
        # Duration distribution by class
        print(f"\n📊 DURATION BY CLASS:")
        for label in sorted(data_df['label'].unique()):
            class_data = data_df[data_df['label'] == label]['audio_length']
            print(f"   {label:10}: mean={class_data.mean():5.2f}s, "
                  f"std={class_data.std():5.2f}s, "
                  f"min={class_data.min():5.2f}s, "
                  f"max={class_data.max():5.2f}s")
        
        # Samples by duration thresholds
        print(f"\n🔍 DURATION THRESHOLD ANALYSIS:")
        short_samples = data_df[data_df['audio_length'] < min_duration]
        long_samples = data_df[data_df['audio_length'] > max_duration]
        valid_samples = data_df[(data_df['audio_length'] >= min_duration) & 
                               (data_df['audio_length'] <= max_duration)]
        
        print(f"   Samples < {min_duration}s: {len(short_samples):4d} ({len(short_samples)/len(data_df)*100:5.1f}%)")
        print(f"   Samples > {max_duration}s: {len(long_samples):4d} ({len(long_samples)/len(data_df)*100:5.1f}%)")
        print(f"   Valid samples ({min_duration}-{max_duration}s): {len(valid_samples):4d} ({len(valid_samples)/len(data_df)*100:5.1f}%)")
        
        # Valid samples by class
        if len(valid_samples) > 0:
            print(f"\n✅ VALID SAMPLES BY CLASS ({min_duration}-{max_duration}s):")
            valid_class_counts = valid_samples['label'].value_counts().sort_index()
            for label, count in valid_class_counts.items():
                total_class = len(data_df[data_df['label'] == label])
                percentage = (count / total_class) * 100
                print(f"   {label:10}: {count:4d}/{total_class:4d} samples ({percentage:5.1f}% of class)")
        
        # Short samples by class
        if len(short_samples) > 0:
            print(f"\n⚠️  SHORT SAMPLES BY CLASS (< {min_duration}s):")
            short_class_counts = short_samples['label'].value_counts().sort_index()
            for label, count in short_class_counts.items():
                total_class = len(data_df[data_df['label'] == label])
                percentage = (count / total_class) * 100
                print(f"   {label:10}: {count:4d}/{total_class:4d} samples ({percentage:5.1f}% of class)")
        
        # Long samples by class
        if len(long_samples) > 0:
            print(f"\n📏 LONG SAMPLES BY CLASS (> {max_duration}s):")
            long_class_counts = long_samples['label'].value_counts().sort_index()
            for label, count in long_class_counts.items():
                total_class = len(data_df[data_df['label'] == label])
                percentage = (count / total_class) * 100
                print(f"   {label:10}: {count:4d}/{total_class:4d} samples ({percentage:5.1f}% of class)")
    
    # Split statistics (if provided)
    if all(df is not None for df in [train_df, val_df, test_df]):
        print(f"\n🔄 TRAIN/VAL/TEST SPLIT STATISTICS:")
        total_split = len(train_df) + len(val_df) + len(test_df)
        print(f"   Train: {len(train_df):4d} samples ({len(train_df)/total_split*100:5.1f}%)")
        print(f"   Val:   {len(val_df):4d} samples ({len(val_df)/total_split*100:5.1f}%)")
        print(f"   Test:  {len(test_df):4d} samples ({len(test_df)/total_split*100:5.1f}%)")
        
        print(f"\n📊 CLASS DISTRIBUTION IN SPLITS:")
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            print(f"   {split_name}:")
            split_counts = split_df['label'].value_counts().sort_index()
            for label, count in split_counts.items():
                percentage = (count / len(split_df)) * 100
                print(f"     {label:10}: {count:3d} samples ({percentage:5.1f}%)")
    
    print("="*70)

def load_dataset_from_hub(dataset_name, split=None):
    """
    Load the dataset from Hugging Face Hub.
    
    Args:
        dataset_name (str): Name of the dataset on the Hub
        split (str, optional): Specific split to load ('train', 'validation', 'test')
    
    Returns:
        Dataset or DatasetDict: The loaded dataset
    """
    from datasets import load_dataset
    
    print(f"Loading dataset '{dataset_name}' from Hugging Face Hub...")
    
    if split:
        dataset = load_dataset(dataset_name, split=split)
        print(f"Loaded {split} split with {len(dataset)} samples")
    else:
        dataset = load_dataset(dataset_name)
        print("Loaded all splits:")
        for split_name, split_data in dataset.items():
            print(f"  - {split_name}: {len(split_data)} samples")
    
    return dataset

# Example usage
if __name__ == "__main__":
    # Create and push dataset to Hub
    dataset = create_and_push_hf_dataset(
        dataset_name="zaghloul2012/egyptian-dialect-audio",  # Replace with your username
        min_duration=2.0,  # Original duration limits as requested
        max_duration=30.0,  # Original duration limits as requested
        push_to_hub=True,
        private=True,
        skip_download=True  # Skip download since files are already in workspace
    )
    
    print("\nDataset created and pushed to Hub!")
    print("You can now load it with:")
    print("from datasets import load_dataset")
    print("dataset = load_dataset('zaghloul2012/egyptian-dialect-audio')")