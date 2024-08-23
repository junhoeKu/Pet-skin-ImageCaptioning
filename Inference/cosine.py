import pandas as pd
import requests
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, PreTrainedTokenizerFast, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def process_images_and_evaluate_similarity(df, encoder_model_name_or_path, batch_size=100, max_new_tokens=150, device=None, resample=True, resample_frac=1.0, filter_keyword=None):
    # Resample the data if needed
    if resample:
        if filter_keyword:
            df = df[~df.apply(lambda row: row.astype(str).str.contains(filter_keyword)).any(axis=1)]
        df = df.sample(frac=resample_frac).reset_index(drop=True)

    # Device setting
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load image processor and tokenizer
    image_processor = ViTImageProcessor.from_pretrained(encoder_model_name_or_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(encoder_model_name_or_path)

    # Load model
    model = VisionEncoderDecoderModel.from_pretrained(encoder_model_name_or_path)
    model.to(device)

    # Pipeline for image captioning
    image_captioner = pipeline(
        "image-to-text", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=max_new_tokens, 
        image_processor=image_processor, 
        device=0 if torch.cuda.is_available() else -1
    )

    # Function to generate caption from an image URL
    def image_caption(url):
        try:
            with Image.open(requests.get(url, stream=True).raw) as img:
                pixel_values = image_processor(images=img, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values.to(device), num_beams=5, max_new_tokens=max_new_tokens)
                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                return generated_text[0]
        except Exception as e:
            print(f"Error processing image {url}: {e}")
            return ""

    # Function to process a batch of data
    def process_batch(df_batch):
        # Apply the image captioning model to each image in the batch
        df_batch['caption_generation'] = [image_caption(url) for url in tqdm(df_batch['images'], desc='Generating Captions')]

        # Prepare data for similarity calculation
        all_text = df_batch['answer'].tolist() + df_batch['caption_generation'].tolist()
        vectorizer = TfidfVectorizer()
        vectorizer.fit(all_text)

        # Transform captions into TF-IDF vectors
        caption_vec = vectorizer.transform(df_batch['answer'])
        caption_gen_vec = vectorizer.transform(df_batch['caption_generation'])

        # Calculate cosine similarity between the original and generated captions
        cosine_scores = cosine_similarity(caption_vec, caption_gen_vec).diagonal()

        # Save the similarity scores in the DataFrame
        df_batch['score'] = cosine_scores

        return df_batch

    # Process the DataFrame in batches
    df_result = pd.DataFrame()
    for i in range(0, len(df), batch_size):
        df_batch = df.iloc[i:i + batch_size].copy()
        df_batch = process_batch(df_batch)
        df_result = pd.concat([df_result, df_batch])

    # Sort the final DataFrame by the score
    df_result = df_result.sort_values('score', ascending=False)

    return df_result

# Example usage:
# encoder_model_name_or_path = "sihoon00/Bitamin_mutimodal"
# df_result = process_images_and_evaluate_similarity(df_0816, encoder_model_name_or_path, resample=True, resample_frac=0.1, filter_keyword='20220725_45')
# pd.set_option('display.max_colwidth', 500)
# print(df_result.head())
