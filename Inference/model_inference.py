from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

def load_model_and_tokenizer(model_name):
    """
    Load the pretrained model and tokenizer.

    Args:
    - model_name (str): The name of the model to load from Hugging Face Hub.

    Returns:
    - model: The loaded model.
    - tokenizer: The loaded tokenizer.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_image_caption(image_url, model, tokenizer, max_new_tokens=150):
    """
    Generate a caption for the given image using the specified model.

    Args:
    - image_url (str): The URL of the image to caption.
    - model: The loaded model for image captioning.
    - tokenizer: The loaded tokenizer associated with the model.
    - max_new_tokens (int): The maximum number of tokens to generate for the caption.

    Returns:
    - result (list): The generated caption for the image.
    """
    image_captioner = pipeline("image-to-text", model=model, tokenizer=tokenizer)
    result = image_captioner(image_url, max_new_tokens=max_new_tokens)
    return result

def main():
    # Model name
    model_name = "sihoon00/multimodla-Bitamin"
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Image URL for captioning
    image_url = 'https://img.lifet.co.kr/07f3f846-edf1-4d24-87e2-10d6930b5794'
    
    # Generate caption for the image
    caption = generate_image_caption(image_url, model, tokenizer)
    
    # Print the result
    print(caption)

if __name__ == "__main__":
    main()
