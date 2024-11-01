# -*- coding: utf-8 -*-
import os
from data_processing.data_utils import read_csv_file, split_data
from machine_learning.ml_utils import train_model, evaluate_model
from nlp.nlp_utils import tokenize_text, pos_tag
from vision.vision_utils import read_image, display_image

def main():
    print("AI Project Starting...")

    # Exemplu de utilizare a func?iilor
    # 1. Procesarea datelor
    print("Loading data...")
    data = read_csv_file('data/sample_data.csv')
    X_train, X_test, y_train, y_test = split_data(data)

    # 2. �nv�?area automat�
    # (Adaug� aici un model de �nv�?are automat� pentru a-l antrena)

    # 3. NLP
    sample_text = "Aceasta este o propozi?ie de test."
    tokens = tokenize_text(sample_text)
    print("Tokens:", tokens)

    # 4. Viziune computerizat�
    # (Adaug� aici codul pentru a citi ?i a afi?a o imagine)

if __name__ == "__main__":
    main()
