import streamlit as st
import json
import pandas as pd

#IMPORTATION DU MODELE PEFT-T5-base

from transformers import T5ForConditionalGeneration, T5Tokenizer

import torch

# Vérifie si un GPU est disponible et le sélectionne, sinon utilise le CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "C:/Users/LRAVOI/OneDrive - Capgemini/Documents/Hackathon_2024/venv/model_sv"

model_test = T5ForConditionalGeneration.from_pretrained(model_name, return_dict = True)

tokenizer_test = T5Tokenizer.from_pretrained(model_name)

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

txt="..."


#AFFICHAGE SUR STREAMLIT
st.title("AirLingo.TEX")
st.header('Original Text :blue[from a JSON Dataset]', divider='rainbow')
st.write("Alike the training, We're going here to choose randomly one of the Original Text from the JSON dataset to be Sum-Up in the next Page")

uploaded_file = st.file_uploader("Choose a JSON dataset file, alike the training dataset")
if uploaded_file is not None:
    json_content = json.load(uploaded_file,)


    df2 = pd.DataFrame.from_dict(json_content, orient='index')

    # Sélection aléatoire d'une ligne du DataFrame
    ligne_aleatoire = df2.sample(n=1)

     # Extraction de l'élément aléatoire et de son uid
    element_aleatoire = ligne_aleatoire['original_text'].iloc[0]
    uid_correspondant = ligne_aleatoire['uid'].iloc[0]

    st.write(f"Le texte Original est pris aléatoirement avec l'uid '{uid_correspondant}'.")
    txt=element_aleatoire



else:
    #1 ENTREZ LE TEXTE
    st.header("OR :blue[input directly the text] you want here", divider='rainbow')
    
    txt = st.text_area(
    "Text to analyze",
    "These general Standard Conditions of Sale apply to any sale of Products and/or Services sold by the Seller to its Customer(s), excluding brokerage or other distributor activities. The purchase of the Products and/or Services by a Customer is considered to be performed within the framework of its professional activities."
    )


#3 VOICI LE RESUME
st.subheader("Here the :red[PEFT_T5-Base sum-up]",divider="red")

summary=summarize_test(txt)
st.write({summary})

#4 EVALUATION
st.subheader("Let see our :green[primary evaluation]",divider="green")

st.caption(f'_You wrote_ :blue[{len(txt)}] _characters._')

if len(txt)>len(summary):
    st.caption(f'_The summary contains_ :red[{len(summary)}] _characters_.:sunglasses: ==> The model fit to the text')
else:
    st.caption(f'_The summary contains_ :red[{len(summary)}] _characters_.:chicken: ==> Something wrong happened here...maybe the text was too short')
  
    
    




