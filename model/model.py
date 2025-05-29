import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import joblib



curr_dir =  os.path.dirname(os.path.abspath(__file__))

def get_data_path():
    # Get the path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    return data_dir


def load_data():
    # Load the dataset
    data_dir = get_data_path()
    df = pd.read_csv(os.path.join(data_dir, 'transactions.csv'))

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Missing values found in the dataset.")
        df = df.dropna()
    else:
        print("No missing values found in the dataset.")
    
    # Check for duplicates
    if df.duplicated().sum() > 0:
        print("Duplicates found in the dataset.")
        df = df.drop_duplicates()
    else:
        print("No duplicates found in the dataset.")
    
    return df

def split_data(df):
    # Split the data into training and testing sets
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Check the ratio of fraud and non-fraud transactions
    print("Fraudulent transactions in the dataset: ", y.value_counts())
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    print("\nTraining Neural Network Model with scikit-learn...\n")

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=100,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on test data
    y_pred = model.predict(X_test)

    print("Model Evaluation:")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save the trained model
    model_path = os.path.join(curr_dir, 'fraud_model_sklearn.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    return model

def main():
    # Load the data
    df = load_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(df)
    
   
    # normalize the train and test ammount ans time columndata that does not have the class label

    # Normalize the 'Amount' and 'Time' columns

    if not {'Time','Amount'}.issubset(X_test.columns):
        raise KeyError('Amount and TIme column is required in the dataset')
    

    scaler = MinMaxScaler()
    X_train_scaled_cols = scaler.fit_transform(X_train[['Time','Amount']])
    X_test_scaled_cols = scaler.transform(X_test[['Time','Amount']])

    # already been excecuted
    #scalerpk = os.path.join(curr_dir, 'scaler.pkl')

    #joblib.dump(scaler,scalerpk)

    X_train[['Time','Amount']] = X_train_scaled_cols
    X_test[['Time','Amount']] = X_test_scaled_cols


    # Save the split data to CSV files
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    model = train_model(X_train, y_train, X_test, y_test)


    modelpkl = os.path.join(curr_dir,'model.pkl')

    joblib.dump(model,modelpkl)










    # train the model
    # model = train_model(X_train, y_train)
    # save the model







if __name__ == "__main__":
    main()