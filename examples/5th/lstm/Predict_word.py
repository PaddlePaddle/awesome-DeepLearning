from tensorflow.keras.models import load_model
import numpy as np
import pickle


model_path = 'model/next_word.h5'
tokenizer_path = 'model/tokenizer.pkl'

model = load_model(model_path, custom_objects=None, compile=False)
tokenizer = pickle.load(open(tokenizer_path, 'rb'))

def Predict_Next_Words(model, tokenizer, text):

    for i in range(4):
        sequence = tokenizer.texts_to_sequences([text])[i]
        sequence = np.array(sequence)
        
        preds = model.predict_classes(sequence)
        predicted_word = ""
        
        for key, value in tokenizer.word_index.items():
            if value == preds:
                predicted_word = key
                break
            
        print(predicted_word)
        
        
while(True):
    text = input("Enter your line: ")
    if text == "stop":
        print("Ending The Program.....")
        break
    else:
        try:
            text = text.split(" ")
            text = text[-1]
            text = ''.join(text)
            Predict_Next_Words(model, tokenizer, text)
        except:
            continue
