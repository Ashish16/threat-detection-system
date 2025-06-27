import csv
import datetime

def save_evaluation_summary(filepath, accuracy, precision, recall, f1, roc_auc):
    fieldnames = ['timestamp', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    try:
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # Write header only if file is empty
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)
    except Exception as e:
        print("Error saving evaluation summary:", e)
