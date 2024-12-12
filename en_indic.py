import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from IndicTransToolkit import IndicProcessor

target_lag = {
    "Hindi" : "hin_Deva",
    "Malayalam" : "mal_Mlym",
    "Tamil" : "tam_Taml",
    "Telugu" : "tel_Telu"
}

def translate_sentence(input_sentences, language):
    model_name = "ai4bharat/indictrans2-en-indic-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

    ip = IndicProcessor(inference=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    model.to(DEVICE)

    translations = []
    translations = ''
    sentences = input_sentences.split('\n')

    for sentence in sentences:
        if sentence != '':
            batch = ip.preprocess_batch(
                [sentence],
                src_lang="eng_Latn",
                tgt_lang=target_lag[language],
            )

            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)

            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            with tokenizer.as_target_tokenizer():
                generated_tokens = tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            translation = ip.postprocess_batch(generated_tokens, lang=target_lag[language])
            print(translation)
            translations += f'{translation[0]}\n'

    return translations