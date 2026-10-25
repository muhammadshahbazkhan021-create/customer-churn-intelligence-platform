import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def _save_and_close(save_path: str = None) -> None:
    """Helper function to handle saving and closing plots."""
    if save_path:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    labels: list = None, 
    normalize: bool = True, 
    save_path: str = None
) -> None:
    """
    Plots a standardized Confusion Matrix using a Seaborn heatmap.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True ground truth labels.
    y_pred : np.ndarray
        Predicted labels from the model.
    labels : list, optional
        List of label names to determine the axes order and tick labels.
    normalize : bool, default=True
        Whether to normalize the counts into proportions.
    save_path : str, optional
        Path to save the generated image (e.g., 'reports/figures/cm.png').
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Format and values assignment
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
        
    # Plotting setup
    plt.figure(figsize=(6, 5))
    xticklabels = labels if labels is not None else 'auto'
    yticklabels = labels if labels is not None else 'auto'
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues', 
        xticklabels=xticklabels, 
        yticklabels=yticklabels,
        cbar=True
    )
    
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save or show plot
    _save_and_close(save_path)


def plot_roc_curve(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    save_path: str = None
) -> None:
    """
    Plots a Receiver Operating Characteristic (ROC) curve with AUC score.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True ground truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    save_path : str, optional
        Path to save the generated image.
    """
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plotting setup
    plt.figure(figsize=(7, 6))
    
    # Plot ROC curve with AUC in legend
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    
    # Dashed diagonal reference line (random guess)
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Save or show plot
    _save_and_close(save_path)


def plot_feature_importance(
    feature_names: list, 
    importances: np.ndarray, 
    top_n: int = 15, 
    save_path: str = None
) -> None:
    """
    Plots a horizontal bar chart of the top N features in descending order.
    
    Parameters:
    -----------
    feature_names : list or np.ndarray
        Names of the features.
    importances : np.ndarray
        Importance scores corresponding to each feature.
    top_n : int, default=15
        Number of top features to display.
    save_path : str, optional
        Path to save the generated image.
    """
    # Convert inputs to numpy arrays for easier indexing
    feature_names = np.array(feature_names)
    importances = np.array(importances)
    
    # Sort features by importance in descending order
    sorted_idx = np.argsort(importances)[::-1]
    
    # Select top N elements
    top_idx = sorted_idx[:top_n]
    top_features = feature_names[top_idx]
    top_scores = importances[top_idx]
    
    # Invert order for plotting from top to bottom in horizontal bar chart
    top_features = top_features[::-1]
    top_scores = top_scores[::-1]
    
    # Plotting setup
    plt.figure(figsize=(10, max(5, top_n * 0.4)))
    
    # Create horizontal bar chart
    bars = plt.barh(top_features, top_scores, color='skyblue', edgecolor='grey', height=0.6)
    
    # Add values on the bars for clarity
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + (max(top_scores) * 0.01), 
            bar.get_y() + bar.get_height()/2, 
            f'{width:.4f}', 
            va='center', 
            ha='left', 
            fontsize=9
        )
        
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {min(top_n, len(feature_names))} Feature Importances', fontsize=14, pad=15)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Save or show plot
    _save_and_close(save_path)
