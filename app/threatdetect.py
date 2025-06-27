import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score
from app.evaluation_app import current_data_evaluation
from scripts.save_data import save_evaluation_summary
from pathlib import Path
def get_and_store_metrics(filepath, y_test, y_pred, y_scores):
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_scores)
    
    # Save summary
    save_evaluation_summary(filepath, accuracy, precision, recall, f1, roc_auc)

def threatDetectFun(seedval):

    # Step 1: Create synthetic dataset (for demonstration purposes)
    np.random.seed(seedval)

    # Generate normal network traffic data
    normal_data = np.random.normal(loc=0, scale=1, size=(1000, 20))

    # Generate abnormal network traffic data (simulating attacks)
    abnormal_data = np.random.normal(loc=5, scale=1, size=(100, 20))

    # Combine the data
    data = np.vstack((normal_data, abnormal_data))
    labels = np.hstack((np.zeros(1000), np.ones(100)))  # 0: normal, 1: anomaly

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels

    # Step 2: Split data into features and labels
    X = df.drop('label', axis=1)
    y = df['label']

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    # Step 4: Initialize and train the Isolation Forest model
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=3)
    model.fit(X_train)

    # Step 5: Make predictions on the test set
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Step 6: Convert predictions to binary labels (1 for anomaly, 0 for normal)
    y_pred_train = np.where(y_pred_train == -1, 1, 0)
    y_pred_test = np.where(y_pred_test == -1, 1, 0)

    # Step 7: Evaluate the model
    print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))
    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, y_pred_test))

    # decision_function gives anomaly scores (higher = more normal)
    scores = model.decision_function(X_test)

    # Invert so higher score = more anomalous
    y_scores = -scores

    current_dir = Path(__file__).resolve().parent.parent
    filepath = current_dir / "data" / "data.csv"
    get_and_store_metrics(filepath, y_test, y_pred_test, y_scores)
    current_data_evaluation( y_test, y_pred_test, y_scores)


if __name__ == "__main__":
    threatDetectFun()