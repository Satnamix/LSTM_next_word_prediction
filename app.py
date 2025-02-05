import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


model=load_model('LSTM RNN Practice/next_word_lstm.h5')
with open('LSTM RNN Practice/tokenizer.pickle','rb') as file:
    tokenizer=pickle.load(file)
    

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    text=text.lower()
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list],maxlen=max_sequence_len)
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Next Word Prediction Using LSTM")
# st.write("Enter")
input_text=st.text_area("Enter the sequence of words","To be not to be")

if st.button("Predict next word"):
    max_sequence_len=model.input_shape[1]
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next Word Prediction:{next_word}")

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Based on Shakespeare Hamlet Dataset<br> Developed with ❤ by <a style='display: block;color: white;text-align: center;' href="https://github.com/Satnamix" target="_blank">Satnam Singh</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)