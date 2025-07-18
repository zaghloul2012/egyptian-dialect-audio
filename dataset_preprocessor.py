import numpy as np
import librosa
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoFeatureExtractor, 
    WhisperFeatureExtractor,
    Wav2Vec2FeatureExtractor,
    # HubertFeatureExtractor is usually imported as Wav2Vec2FeatureExtractor
)
import gc


def preprocess_audio_dataset(
    dataset_name='zaghloul2012/egyptian-dialect-audio',
    model_id="Seyfelislem/whisper-medium-arabic",
    model_type="whisper",  # "whisper", "wav2vec2", "hubert", or "auto"
    sampling_rate=None,  # Will use model's default if None
    max_duration=10,
    batch_size=16,
    num_proc=1,
    remove_columns=None
):
    """
    Universal preprocessing function for audio classification datasets.
    Works with Whisper, Wav2Vec2, HuBERT, and other audio models.
    
    Args:
        dataset_name (str): Name of the dataset on HuggingFace Hub
        model_id (str): Pre-trained model ID for feature extraction
        model_type (str): Type of model ("whisper", "wav2vec2", "hubert", "auto")
        sampling_rate (int): Target sampling rate (uses model default if None)
        max_duration (int): Maximum audio duration in seconds
        batch_size (int): Batch size for processing
        num_proc (int): Number of processes for parallel processing
        remove_columns (list): Additional columns to remove
    
    Returns:
        tuple: (processed_dataset, feature_extractor, label_mappings, model_info)
    """
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_name)
    
    # Auto-detect model type if needed
    if model_type == "auto":
        model_id_lower = model_id.lower()
        if "whisper" in model_id_lower:
            model_type = "whisper"
        elif "wav2vec2" in model_id_lower:
            model_type = "wav2vec2"
        elif "hubert" in model_id_lower:
            model_type = "hubert"
        else:
            print("Warning: Could not auto-detect model type. Defaulting to 'wav2vec2'")
            model_type = "wav2vec2"
    
    # Load the appropriate feature extractor
    print(f"Loading {model_type} feature extractor...")
    if model_type == "whisper":
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    elif model_type in ["wav2vec2", "hubert"]:
        # HuBERT uses the same feature extractor as Wav2Vec2
        try:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        except:
            # Fallback to AutoFeatureExtractor
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    else:
        # Generic approach for other models
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    
    # Set sampling rate
    if sampling_rate is None:
        sampling_rate = getattr(feature_extractor, 'sampling_rate', 16000)
    
    print(f"Using sampling rate: {sampling_rate} Hz")
    
    # Get unique labels and create mappings
    print("Creating label mappings...")
    all_labels = set()
    for split in dataset:
        all_labels.update(dataset[split]['label'])
    
    unique_labels = sorted(list(all_labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    
    def load_and_resample_audio(example):
        """Load audio and resample to target sampling rate"""
        try:
            # The audio is already loaded in the dataset
            audio_array = example['audio']['array']
            current_sr = example['audio']['sampling_rate']
            
            # Resample if necessary
            if current_sr != sampling_rate:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=current_sr, 
                    target_sr=sampling_rate
                )
            
            # Convert to float32 and ensure it's 1D
            audio_array = np.array(audio_array, dtype=np.float32)
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()
            
            return {
                'audio': audio_array,
                'target_label': label2id[example['label']]  # Use different name to avoid conflicts
            }
        except Exception as e:
            print(f"Error processing audio: {e}")
            # Return empty audio array if processing fails
            return {
                'audio': np.zeros(int(sampling_rate * 0.1), dtype=np.float32),
                'target_label': label2id[example['label']]
            }
    
    def preprocess_function_whisper(examples):
        """Feature extraction for Whisper models"""
        try:
            audio_arrays = [audio for audio in examples["audio"]]
            
            # Filter and pad short audio
            processed_audio = []
            for audio in audio_arrays:
                if len(audio) < int(sampling_rate * 0.1):
                    min_length = int(sampling_rate * 0.1)
                    audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
                processed_audio.append(audio)
            
            # Whisper feature extraction
            features = feature_extractor(
                processed_audio,
                sampling_rate=sampling_rate,
                max_length=int(sampling_rate * max_duration),
                truncation=True,
                padding="max_length",
                return_attention_mask=False,
                return_tensors="np"
            )
            
            return {
                "input_features": features["input_features"],
                "label": examples["target_label"]
            }
            
        except Exception as e:
            print(f"Error in Whisper feature extraction: {e}")
            batch_size = len(examples["audio"])
            # Whisper mel-spectrogram features (80 mel bins)
            dummy_features = np.zeros((batch_size, 80, 3000))
            return {
                "input_features": dummy_features,
                "label": examples["target_label"]
            }
    
    def preprocess_function_wav2vec(examples):
        """Feature extraction for Wav2Vec2/HuBERT models"""
        try:
            audio_arrays = [audio for audio in examples["audio"]]
            
            # Filter and pad short audio
            processed_audio = []
            for audio in audio_arrays:
                if len(audio) < int(sampling_rate * 0.1):
                    min_length = int(sampling_rate * 0.1)
                    audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
                processed_audio.append(audio)
            
            # Wav2Vec2/HuBERT feature extraction
            max_length = int(sampling_rate * max_duration)
            
            features = feature_extractor(
                processed_audio,
                sampling_rate=sampling_rate,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="np"
            )
            
            return {
                "input_values": features["input_values"],
                "label": examples["target_label"]
            }
            
        except Exception as e:
            print(f"Error in Wav2Vec2/HuBERT feature extraction: {e}")
            batch_size = len(examples["audio"])
            max_length = int(sampling_rate * max_duration)
            dummy_features = np.zeros((batch_size, max_length))
            return {
                "input_values": dummy_features,
                "label": examples["target_label"]
            }
    
    # Select the appropriate preprocessing function
    if model_type == "whisper":
        preprocess_function = preprocess_function_whisper
        input_column = "input_features"
    else:  # wav2vec2, hubert, or others
        preprocess_function = preprocess_function_wav2vec
        input_column = "input_values"
    
    # Process each split
    processed_dataset = {}
    
    for split_name in dataset.keys():
        print(f"Processing {split_name} split...")
        
        # First, resample audio and convert labels
        split_data = dataset[split_name].map(
            load_and_resample_audio,
            num_proc=num_proc,
            desc=f"Resampling {split_name} audio"
        )
        
        # Remove unnecessary columns
        columns_to_remove = [col for col in split_data.column_names 
                           if col not in ['audio', 'target_label']]
        if remove_columns:
            columns_to_remove.extend(remove_columns)
        
        if columns_to_remove:
            # Only remove columns that actually exist
            existing_columns = [col for col in columns_to_remove if col in split_data.column_names]
            if existing_columns:
                split_data = split_data.remove_columns(existing_columns)
        
        # Apply feature extraction
        split_data = split_data.map(
            preprocess_function,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=['audio'],
            desc=f"Extracting features for {split_name}"
        )
        
        processed_dataset[split_name] = split_data
        
        print(f"Processed {split_name}: {len(split_data)} samples")
        if len(split_data) > 0:
            sample_feature = split_data[0][input_column]
            try:
                if hasattr(sample_feature, 'shape'):
                    sample_shape = sample_feature.shape
                else:
                    # Handle case where it's a list or other type
                    sample_shape = np.array(sample_feature).shape
                print(f"Feature shape: {sample_shape}")
            except Exception as e:
                print(f"Could not determine feature shape: {e}")
        
        # Clean up memory
        gc.collect()
    
    # Create the final dataset dict
    processed_dataset = DatasetDict(processed_dataset)
    
    # Model information
    model_info = {
        'model_type': model_type,
        'model_id': model_id,
        'sampling_rate': sampling_rate,
        'max_duration': max_duration,
        'input_column': input_column,
        'feature_extractor_type': type(feature_extractor).__name__
    }
    
    # Print summary
    print("\nDataset preprocessing completed!")
    print(f"Model type: {model_type}")
    print(f"Model ID: {model_id}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Max duration: {max_duration} seconds")
    print(f"Input column: {input_column}")
    print(f"Label mappings: {label2id}")
    
    for split_name, split_data in processed_dataset.items():
        print(f"{split_name}: {len(split_data)} samples")
    
    return processed_dataset, feature_extractor, (label2id, id2label), model_info

# Convenience functions for specific model types
def preprocess_for_whisper(dataset_name, model_id="openai/whisper-base", **kwargs):
    """Convenience function for Whisper models"""
    return preprocess_audio_dataset(
        dataset_name=dataset_name,
        model_id=model_id,
        model_type="whisper",
        **kwargs
    )

def preprocess_for_wav2vec2(dataset_name, model_id="facebook/wav2vec2-base", **kwargs):
    """Convenience function for Wav2Vec2 models"""
    return preprocess_audio_dataset(
        dataset_name=dataset_name,
        model_id=model_id,
        model_type="wav2vec2",
        **kwargs
    )

def preprocess_for_hubert(dataset_name, model_id="facebook/hubert-base-ls960", **kwargs):
    """Convenience function for HuBERT models"""
    return preprocess_audio_dataset(
        dataset_name=dataset_name,
        model_id=model_id,
        model_type="hubert",
        **kwargs
    )


# Example usage - Test one at a time
if __name__ == "__main__":
    print()
    
    # print("=== Example 1: Whisper ===")
    # try:
    #     dataset_whisper, fe_whisper, (l2i_w, i2l_w), info_w = preprocess_for_whisper(
    #         dataset_name='zaghloul2012/egyptian-dialect-audio',
    #         model_id="Seyfelislem/whisper-medium-arabic",
    #         max_duration=10  # Limit duration for faster processing
    #     )
    #     print("✅ Whisper preprocessing completed successfully!")
    #     print(f"Train samples: {len(dataset_whisper['train'])}")
    #     print(f"Label mappings: {l2i_w}")
    # except Exception as e:
    #     print(f"❌ Error with Whisper: {e}")
    
    # Uncomment one at a time to test
    # print("\n=== Example 2: Wav2Vec2 ===")
    # try:
    #     dataset_wav2vec2, fe_wav2vec2, (l2i_w2v, i2l_w2v), info_w2v = preprocess_for_wav2vec2(
    #         'zaghloul2012/egyptian-dialect-audio',
    #         model_id="arbml/wav2vec2-large-xlsr-53-arabic-egyptian",
    #         max_duration=10
    #     )
    #     print("✅ Wav2Vec2 preprocessing completed successfully!")
    # except Exception as e:
    #     print(f"❌ Error with Wav2Vec2: {e}")
    
    # print("\n=== Example 3: HuBERT ===")
    # try:
    #     dataset_hubert, fe_hubert, (l2i_h, i2l_h), info_h = preprocess_for_hubert(
    #         'zaghloul2012/egyptian-dialect-audio',
    #         model_id="omarxadel/hubert-large-arabic-egyptian",
    #         max_duration=10
    #     )
    #     print("✅ HuBERT preprocessing completed successfully!")
    # except Exception as e:
    #     print(f"❌ Error with HuBERT: {e}")