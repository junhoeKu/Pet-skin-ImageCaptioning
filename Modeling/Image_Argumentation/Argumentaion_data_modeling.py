import Image_Flip
import requests
import os
from datasets import Dataset, DatasetDict, concatenate_datasets, Features, Value
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
import pandas as pd
from tqdm import tqdm
import nltk
import metrics
import evaluate


# Image download function
def download_image(url, save_path):
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

# text preprocessing step
def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(captions, padding="max_length", max_length=max_target_length)
    return labels

# image preprocessing step
def feature_extraction_fn(image_paths, check_image=True):
    temp_dir = 'image'
    os.makedirs(temp_dir, exist_ok=True)

    images = []
    to_keep = []
    for image_url in image_paths:
        try:
            image_file = download_image(image_url, os.path.join(temp_dir, os.path.basename(image_url)))
            if image_file:
                count_num = Image_Flip.check_and_add_image(image_file)
                img = Image_Flip.tile_image(image_file,count_num)
                # img = Image.open(image_file).convert("RGB")
                images.append(img)
                to_keep.append(True)
            else:
                to_keep.append(False)
        except Exception:
            to_keep.append(False)
    image_paths = [image_paths[i] for i in range(len(image_paths)) if to_keep[i]]

    encoder_inputs = feature_extractor(images=images, return_tensors="np")
    return encoder_inputs.pixel_values, to_keep

# Preprocessing function
def preprocess_fn(examples, max_target_length, check_image=True):
    image_paths = examples['image_path']
    captions = examples['caption']

    model_inputs = {}
    model_inputs['pixel_values'],to_keep = feature_extraction_fn(image_paths, check_image=check_image)

    captions = [captions[i] for i in range(len(captions)) if to_keep[i]]

    model_inputs['labels'] = tokenization_fn(captions, max_target_length).input_ids

    return model_inputs

# Function to process dataset in chunks
def process_in_chunks(dataset, batch_size, max_target_length):
    processed_chunks = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        chunk = dataset.select(range(i, min(i + batch_size, len(dataset))))
        processed_chunk = chunk.map(
            function=preprocess_fn,
            batched=True,
            fn_kwargs={"max_target_length": max_target_length},
            remove_columns=chunk.column_names
        )
        processed_chunks.append(processed_chunk)
    return concatenate_datasets(processed_chunks)

def main(file_path, tokenizer, feature_extractor):
    df = file_path

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.rename(columns={'answer': 'caption', 'images': 'image_path'})

    # Create a Dataset from the dataframe
    ds = Dataset.from_pandas(df)

    # Split the dataset into training and test sets
    train_test_split = ds.train_test_split(test_size=0.1)

    # DatasetDict creation
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

    # Define features for the dataset
    features = Features({
        'image_path': Value('string'),
        'caption': Value('string')
    })

    # Cast the dataset with the new features
    ds_1 = dataset_dict.cast(features)

    image_link_list = []

    # Batch size for processing
    batch_size = 150  # Adjust based on your system capacity

    # Process train and test datasets separately in chunks
    processed_train = process_in_chunks(ds_1['train'], batch_size=batch_size, max_target_length=1024)
    processed_test = process_in_chunks(ds_1['test'], batch_size=batch_size, max_target_length=1024)

    # Create a new DatasetDict with the processed datasets
    combined_dataset = DatasetDict({
        'train': processed_train,
        'test': processed_test
    })

    return combined_dataset

def training(output_dir, checkpoint_path, dataset):

    # If the dataset was uploaded as a DatasetDict, you should already have train and test splits
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # GPT2 only has bos/eos tokens but not decoder_start/pad tokens
    tokenizer.pad_token = tokenizer.eos_token

    # update the model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        output_dir=output_dir,
        generation_max_length=100,  # 생성할 문장의 최대 길이 설정
        generation_num_beams=4,
        num_train_epochs=5,
        save_steps=5000,
        save_total_limit=2,
        push_to_hub=True,  # 허브에 업로드를 활성화
        hub_model_id="",  # 허브 리포지토리 이름
        hub_strategy=os.path.join(checkpoint_path,"checkpoint"),  # 체크포인트마다 업로드
        hub_token=""  # 허깅페이스 토큰
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        nltk.download("punkt", quiet=True)


    file_path = ''  # data_root
    save_data_path = ''
    df = pd.read_csv(os.path.join(file_path,'aug_final_df_0816.csv'))[['images','answer']]

    image_model = "ddobokki/vision-encoder-decoder-vit-gpt2-coco-ko"
    feature_extractor = AutoFeatureExtractor.from_pretrained(image_model)
    tokenizer = AutoTokenizer.from_pretrained(image_model)
    model = VisionEncoderDecoderModel.from_pretrained(image_model)

    combined_dataset = main(df, tokenizer, feature_extractor)
    print("\nTraining Start.")
    training('','./checkpoint',combined_dataset)
    print("\nTraining Complete.")
