from src.models import logistic_reg_model
from src.processed_data import processed_dataset
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
        
    X,y = processed_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
    logistic_model = logistic_reg_model(X_train, y_train, X_test, y_test)
    roc, acc, f1, cm = evaluate_model(logistic_model, X_test, y_test)

    print("Final Test Metrics:")
    print("ROC-AUC:", roc)
    print("Accuracy:", acc)
    print("F1:", f1)
    print("Confusion Matrix:\n", cm)
    print(logistic_model)