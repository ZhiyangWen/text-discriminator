import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
from extract_feature import read_extract

print("is running")

def main():
    train_df = pd.read_csv("data/splits/train.csv")
    validation_df = pd.read_csv("data/splits/validation.csv")
    test_df = pd.read_csv("data/splits/test.csv")

    X_train, y_train = read_extract(train_df)
    X_validation, y_validation = read_extract(validation_df)
    X_test, y_test = read_extract(test_df)

    clf =  RandomForestClassifier(n_estimators=100, random_state=42) #create model, can only train on numeric features
    clf.fit(X_train, y_train) #train the model, the model learns pattern

    print("Validation Results")
    y_val_predict = clf.predict(X_validation)
    print(classification_report(y_validation, y_val_predict, digits=4))

    print("Test Results")
    y_test_predict = clf.predict(X_test) #use the model to make predictions
    print(classification_report(y_test, y_test_predict, digits=4)) #summarize the overall outcome

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/rf_discriminator.joblib")
    print("Saved model to models/rf_discriminator.joblib")

if __name__ == "__main__":
    main()