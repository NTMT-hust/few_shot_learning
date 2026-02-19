from StratifiedKFoldCrossValidation import StratifiedKFoldCrossValidation
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import os
if __name__ == '__main__':
    dataset_path = "/kaggle/input/mcrgcn-imaged-0-1/ResultTest4/dataset"
    
    model = StratifiedKFoldCrossValidation(
        model_name="EfficientNetB1Classifier",
        dataset_path=dataset_path,
        k_folds=5,
        num_epochs=100,
        freeze_epochs=20,
        batch_size=32,
        lr=0.00015,
    )

    fold_results, fold_models, class_names = model.run()

    print(f'\n{"="*60}')
    print('FINAL SUMMARY')
    print(f'{"="*60}')
    print(f'Classes: {class_names}')
    print(f'Number of folds: {len(fold_results)}')

    # Visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    for i, result in enumerate(fold_results):
        history = result['history']

        axes[0, 0].plot(history['train_loss'], label=f"Fold {i+1}", alpha=0.7)
        axes[0, 1].plot(history['val_loss'], label=f"Fold {i+1}", alpha=0.7)
        axes[0, 2].plot(history['train_f1'], label=f"Fold {i+1}", alpha=0.7)
        axes[1, 0].plot(history['train_acc'], label=f"Fold {i+1}", alpha=0.7)
        axes[1, 1].plot(history['val_acc'], label=f"Fold {i+1}", alpha=0.7)
        axes[1, 2].plot(history['val_f1'], label=f"Fold {i+1}", alpha=0.7)
        axes[2, 0].plot(history['train_auc'], label=f"Fold {i+1}", alpha=0.7)
        axes[2, 1].plot(history['val_auc'], label=f"Fold {i+1}", alpha=0.7)
        axes[2, 2].plot(history['val_sens'], label=f"Fold {i+1}", alpha=0.7)

    axes[0, 0].set_title('Training Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 2].set_title('Training F1-Score')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 2].set_title('Validation F1-Score')
    axes[2, 0].set_title('Training AUC')
    axes[2, 1].set_title('Validation AUC')
    axes[2, 2].set_title('Validation Sensitivity')

    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('imbalanced_kfold_comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n✓ Visualization saved as 'imbalanced_kfold_comprehensive_results.png'")
