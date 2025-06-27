import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def generate_dummy_data(seed=42, n_samples=100):
    np.random.seed(seed)
    y_test = np.random.choice([0, 1], size=n_samples)
    y_pred = np.random.choice([0, 1], size=n_samples)
    y_scores = np.random.rand(n_samples)
    return y_test, y_pred, y_scores

def display_metrics(y_test, y_pred):
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    accuracy = (y_test == y_pred).mean()
    st.metric("Accuracy", f"{accuracy:.2%}")
    st.write("### Classification Report")
    st.dataframe(pd.DataFrame(report_dict).transpose())
    st.markdown("""
    **Metric Explanation:**  
    - **Accuracy:** Overall correctness of the model.  
    - **Precision:** Correct positive predictions / Total predicted positives.  
    - **Recall:** Correct positive predictions / Total actual positives.  
    - **F1-score:** Harmonic mean of precision and recall.
    """)
    gap()

def plot_confusion_matrix(y_test, y_pred):
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    # ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    plt.close(fig)

def describe_confusion_matrix():
    st.markdown("""
        ##### True Negatives  
        **Position:** (0, 0)  
        **Meaning:** ✅ The model accurately identified all negatives. 
        ##### False Positives  
        **Position:** (0, 1)  
        **Meaning:** ⚠️ The model falsely flagged negatives as positives.
        ##### False Negatives  
        **Position:** (1, 0)  
        **Meaning:** ⚠️ The model failed to detect actual positives.
        ##### True Positives  
        **Position:** (1, 1)  
        **Meaning:** ✅ The model correctly detected all positives.
    """)



def plot_roc_curve(y_test, y_scores):
    st.write("### ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("""
    - Shows the tradeoff between correctly detecting positives (Recall) and incorrectly flagging negatives.  
    - Higher AUC indicates better model performance.
    """)

def plot_precision_recall_curve(y_test, y_scores):
    st.write("### Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    # ax.set_title('Precision-Recall Curve')
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("""
    - Illustrates the balance between precision and recall for different decision thresholds.  
    - Important when positive class is rare or costly to miss.
    """)

def show_sample_predictions(y_test, y_pred, y_scores, n=10):
    sample_df = pd.DataFrame({
        "True Label": y_test[:n],
        "Predicted Label": y_pred[:n],
        "Anomaly Score": y_scores[:n]
    })
    st.write("### Sample Predictions")
    st.table(sample_df)
    st.markdown("""
    **Sample Predictions Explanation:**  
    - Shows example data points with their actual and predicted classes.  
    - Anomaly score indicates how confident the model is about an anomaly.
    """)

def show_evaluation_history(filepath):
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            st.info("No evaluation records found.")
            return
        st.write("### Evaluation History")
        # st.dataframe(df) //instead of showing all details, we will have pagination

        # Sort by timestamp if needed
        df = df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)

        # Pagination setup
        rows_per_page = 5
        total_rows = len(df)
        total_pages = (total_rows - 1) // rows_per_page + 1

        page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

        start_idx = (page_number - 1) * rows_per_page
        end_idx = start_idx + rows_per_page

        # Show current page of data
        st.dataframe(df.iloc[start_idx:end_idx])


        fig = px.line(df, x='timestamp', y=['accuracy', 'f1', 'roc_auc'])
        fig.update_yaxes(range=[0.85, 1.0])  # zoom y-axis to 0.85-1.0
        st.plotly_chart(fig)
        # st.line_chart(df.set_index('timestamp')[['accuracy', 'f1', 'roc_auc']])
    except Exception as e:
        st.warning(f"No records found or error reading file: {e}")

def gap():
    st.markdown(
        """
    
        ---
    
        """
    )

def current_data_evaluation( y_test, y_pred, y_scores):
    st.title("Anomaly Detection Model Evaluation")
    display_metrics(y_test, y_pred)
    col1, col2 = st.columns(2)

    with col1:
        plot_confusion_matrix(y_test, y_pred)
    with col2:
        describe_confusion_matrix()

    gap()

    col3, col4 = st.columns(2)
    with col3:
        plot_roc_curve(y_test, y_scores)

    with col4:
        plot_precision_recall_curve(y_test, y_scores)

    gap()
    # plot_confusion_matrix(y_test, y_pred)
    # plot_roc_curve(y_test, y_scores)
    # plot_precision_recall_curve(y_test, y_scores)
    show_sample_predictions(y_test, y_pred, y_scores)


if __name__ == "__main__":
    current_data_evaluation()
