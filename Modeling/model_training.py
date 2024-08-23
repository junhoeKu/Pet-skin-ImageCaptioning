import os
import nltk
import numpy as np
import datasets
import evaluate

from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor

def setup_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        nltk.download("punkt", quiet=True)

def load_model_and_tokenizer(image_encoder_model):
    feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
    tokenizer = AutoTokenizer.from_pretrained(image_encoder_model)
    model = VisionEncoderDecoderModel.from_pretrained(image_encoder_model)
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Update the model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer, feature_extractor

def save_model_and_tokenizer(model, feature_extractor, tokenizer, output_dir):
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def compute_metrics(eval_preds, tokenizer, metric, ignore_pad_token_for_loss):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = safe_batch_decode(tokenizer, preds, skip_special_tokens=True)
    
    if ignore_pad_token_for_loss:
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_labels = safe_batch_decode(tokenizer, labels, skip_special_tokens=True)
    
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels
#토크나이저에 존재하는 토큰만 사용하여 디코드
def safe_batch_decode(tokenizer, sequences, skip_special_tokens=True):
    decoded_texts = []
    for seq in sequences:
        try:
            if isinstance(seq, np.ndarray) or isinstance(seq, list):
                seq = [int(x) for x in seq]
            max_token_id = tokenizer.vocab_size - 1
            valid_seq = [id for id in seq if 0 <= id <= max_token_id]
            decoded_texts.append(tokenizer.decode(valid_seq, skip_special_tokens=skip_special_tokens))
        except Exception as e:
            print(f"Error decoding sequence {seq}: {e}")
            decoded_texts.append("")  
    return decoded_texts
#모델 학습
def train_model(model, tokenizer, feature_extractor, train_dataset, test_dataset, output_dir, hub_token):
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        output_dir=output_dir,
        generation_max_length=100,
        generation_num_beams=4,
        num_train_epochs=5,
        save_steps=5000,
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id="Your_huggingpace",
        hub_strategy="checkpoint",
        hub_token=hub_token
    )

    metric = evaluate.load("rouge")
    ignore_pad_token_for_loss = True

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, metric, ignore_pad_token_for_loss),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
def main():
    setup_nltk()
    
    image_encoder_model = "ddobokki/vision-encoder-decoder-vit-gpt2-coco-ko"
    output_dir = "./image-captioning-output"
    hub_token = "Your API"
    
    model, tokenizer, feature_extractor = load_model_and_tokenizer(image_encoder_model)
    
    # Assuming train_dataset and test_dataset are loaded or prepared elsewhere
    # train_dataset = ...
    # test_dataset = ...

    save_model_and_tokenizer(model, feature_extractor, tokenizer, output_dir)
    
    train_model(model, tokenizer, feature_extractor, train_dataset, test_dataset, output_dir, hub_token)

if __name__ == "__main__":
    main()
