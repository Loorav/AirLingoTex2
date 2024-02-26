#SPECIAL STREAMLIT - 
import streamlit as st

from transformers import T5ForConditionalGeneration, T5Tokenizer

model_test = T5ForConditionalGeneration.from_pretrained(r"C:\Users\LRAVOI\OneDrive - Capgemini\Documents\Hackathon_2024\venv\model_sv", return_dict = True)

tokenizer_test = T5Tokenizer.from_pretrained(r"C:\Users\LRAVOI\OneDrive - Capgemini\Documents\Hackathon_2024\venv\model_sv")

def summarize_test(text):
    inputs = tokenizer_test(text,
                            max_length=512,
                            truncation=True,
                            padding="max_length",
                            return_attention_mask=True,
                            add_special_tokens=True,
                            return_tensors="pt").to(device)
    summarized_ids = model_test.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    ).to(device)
    

    return " ".join([tokenizer_test.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for token_ids in summarized_ids])
