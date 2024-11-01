import pandas as pd
from sklearn.model_selection import train_test_split

def read_csv_file(file_path):
    """Citește un fișier CSV și returnează un DataFrame."""
    return pd.read_csv(file_path)

def split_data(data, test_size=0.2, random_state=42):
    """
    Împarte datele în seturi de antrenament și testare.
    """
    X = data.drop(columns=['target'])  # Asigură-te că 'target' este coloana țintă
    y = data['target']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
