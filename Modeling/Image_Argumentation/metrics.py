ignore_pad_token_for_loss = True
metric = evaluate.load("rouge")

def safe_batch_decode(tokenizer, sequences, skip_special_tokens=True):
    decoded_texts = []
    for seq in sequences:
        try:
            if isinstance(seq, np.ndarray) or isinstance(seq, list):
                # 토큰 ID가 올바른 정수형 타입인지 확인하고, 그렇지 않으면 변환
                seq = [int(x) for x in seq]
            # 범위 내의 토큰 ID만 필터링
            max_token_id = tokenizer.vocab_size - 1
            valid_seq = [id for id in seq if 0 <= id <= max_token_id]
            # 안전하게 디코드
            decoded_texts.append(tokenizer.decode(valid_seq, skip_special_tokens=skip_special_tokens))
        except Exception as e:
            print(f"Error decoding sequence {seq}: {e}")
            decoded_texts.append("")  # 오류 발생 시 빈 문자열로 처리
    return decoded_texts

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # 안전한 디코딩 함수 사용
    decoded_preds = safe_batch_decode(tokenizer, preds, skip_special_tokens=True)

    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # 안전한 디코딩 함수 사용
    decoded_labels = safe_batch_decode(tokenizer, labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result