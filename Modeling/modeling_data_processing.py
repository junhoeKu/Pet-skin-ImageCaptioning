import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, Features, Value
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import os
import requests
from PIL import Image

def load_and_preprocess_dataframe(file_path, drop_keyword):
    # Load the dataframe and preprocess
    df = pd.read_csv(file_path)[['images', 'answer']]
    df = df[~df.apply(lambda row: row.astype(str).str.contains(drop_keyword)).any(axis=1)]
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.rename(columns={'answer': 'caption', 'images': 'image_path'})
    return df

def create_dataset_from_dataframe(df):
    # Create a Dataset from the dataframe
    ds = Dataset.from_pandas(df)
    # Split the dataset into training and test sets
    train_test_split = ds.train_test_split(test_size=0.1)
    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })
    return dataset_dict

def define_dataset_features():
    # Define features for the dataset
    features = Features({
        'image_path': Value('string'),
        'caption': Value('string')
    })
    return features

def download_image(url, save_path):
    # Image download function
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return save_path
        else:
            print(f"Failed to download image from {url}")
            return None
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def tokenization_fn(captions, max_target_length, tokenizer):
    """Run tokenization on captions."""
    labels = tokenizer(captions, padding="max_length", max_length=max_target_length).input_ids
    return labels

def feature_extraction_fn(image_paths, feature_extractor, check_image=True):
    # Image preprocessing step
    temp_dir = '/mnt/data/temp_images'
    os.makedirs(temp_dir, exist_ok=True)

    images = []
    to_keep = []
    for image_url in image_paths:
        try:
            image_file = download_image(image_url, os.path.join(temp_dir, os.path.basename(image_url)))
            if image_file:
                img = Image.open(image_file).convert("RGB")
                images.append(img)
                to_keep.append(True)
            else:
                to_keep.append(False)
        except Exception:
            to_keep.append(False)
    image_paths = [image_paths[i] for i in range(len(image_paths)) if to_keep[i]]

    encoder_inputs = feature_extractor(images=images, return_tensors="np")
    return encoder_inputs.pixel_values

def preprocess_fn(examples, max_target_length, tokenizer, feature_extractor, check_image=True):
    # Preprocessing function
    image_paths = examples['image_path']
    captions = examples['caption']    

    model_inputs = {}
    model_inputs['labels'] = tokenization_fn(captions, max_target_length, tokenizer)
    model_inputs['pixel_values'] = feature_extraction_fn(image_paths, feature_extractor, check_image=check_image)

    return model_inputs

def process_in_chunks(dataset, batch_size, max_target_length, tokenizer, feature_extractor):
    # Function to process dataset in chunks
    processed_chunks = []
    for i in range(0, len(dataset), batch_size):
        chunk = dataset.select(range(i, min(i + batch_size, len(dataset))))
        processed_chunk = chunk.map(
            function=preprocess_fn,
            batched=True,
            fn_kwargs={"max_target_length": max_target_length, "tokenizer": tokenizer, "feature_extractor": feature_extractor},
            remove_columns=chunk.column_names
        )
        processed_chunks.append(processed_chunk)
    return concatenate_datasets(processed_chunks)

def main(file_path, drop_keyword, tokenizer, feature_extractor, batch_size=150, max_target_length=1024):
    # Load and preprocess the dataframe
    df = load_and_preprocess_dataframe(file_path, drop_keyword)
    
    # Create a DatasetDict
    dataset_dict = create_dataset_from_dataframe(df)
    
    # Define features and cast dataset
    features = define_dataset_features()
    ds_1 = dataset_dict.cast(features)
    
    # Process train and test datasets separately in chunks
    processed_train = process_in_chunks(ds_1['train'], batch_size=batch_size, max_target_length=max_target_length, tokenizer=tokenizer, feature_extractor=feature_extractor)
    processed_test = process_in_chunks(ds_1['test'], batch_size=batch_size, max_target_length=max_target_length, tokenizer=tokenizer, feature_extractor=feature_extractor)
    
    # Create a new DatasetDict with the processed datasets
    combined_dataset = DatasetDict({
        'train': processed_train,
        'test': processed_test
    })

    print(combined_dataset)
    return combined_dataset

if __name__ == "__main__":
    # Define your paths and objects here
    file_path = '/kaggle/input/text-aug-data/aug_final_df_0816.csv'
    drop_keyword = '20220725_45'
    
    # You need to define or load your tokenizer and feature_extractor
    tokenizer = AutoTokenizer.from_pretrained("sihoon00/multimodla-Bitamin")
    image_encoder_model = "ddobokki/vision-encoder-decoder-vit-gpt2-coco-ko"
    feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)  

    combined_dataset = main(file_path, drop_keyword, tokenizer, feature_extractor)
