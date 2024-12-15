import tensorflow as tf
import string
import requests
import pandas as pd


"""

    Получения исходных данных для обучения

"""

response = requests.get('https://raw.githubusercontent.com/laxmimerit/poetry-data/master/adele.txt') # Получение словаря
data=response.text.splitlines()

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences


"""
    Токенизация словаря для последующего создания Embedding слоя.

    (Назначение каждому слову порядкового номера для последующего обучения)

"""
tokenizer=Tokenizer() 
tokenizer.fit_on_texts(data)
encoded_text=tokenizer.texts_to_sequences(data)
wc=tokenizer.word_counts
wi=tokenizer.word_index
vocab_size=len(tokenizer.word_counts)+1 
data_list=[]
for i in encoded_text:
    if len(i)>1:
        for j in range(2,len(i)):
            data_list.append(i[:j])



max_length=20
sequences=pad_sequences(data_list,maxlen=max_length,padding="pre") 
X=sequences[:,:-1]
y=sequences[:,-1]
y=to_categorical(y,num_classes=vocab_size)
seq_length=X.shape[1]

"""

    Создание модели нейронной сети
    1 Слой - Эмбенддинг
    2 Слой - LSTM
    3 Слой - Полносвязный

"""

model=Sequential()
model.add(Embedding(vocab_size,50,input_length=seq_length)) 

model.add(LSTM(100,return_sequences=True))
model.add(Dense(vocab_size,activation="softmax")) #

model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


text_lenght= 15 # Кол-во слов в строке


"""

    Функция генерации текста, на вход принимается строка из которой небходимо построить связанное предложение (input_text) и кол-во предложений, неоюходимых для генерации (no_lines)


"""

def generate_text(input_text, no_lines):
    general_text=[]
    for i in range(no_lines):
        text=[]
        for _ in range(text_lenght):
            encoded=tokenizer.texts_to_sequences([input_text])
            encoded=pad_sequences(encoded,maxlen=seq_length,padding="pre")
            y_pred=np.argmax(model.predict(encoded),axis=-1) 
            
            predicted_word=""
            for word,index in tokenizer.word_index.items():
                if index==y_pred:
                    predicted_word=word
                    break
                    
            input_text=input_text +' '+ predicted_word
            text.append(predicted_word)
        
        input_text=text[-1]
        text=" ".join(text) # input text will be the last word of first created line
        general_text.append(text)
    
    return general_text

input_text="i want to see you"
text_produced=generate_text(input_text,6)
print(text_produced)


input_text="hello"
text_produced=generate_text(input_text,6)
print(text_produced)