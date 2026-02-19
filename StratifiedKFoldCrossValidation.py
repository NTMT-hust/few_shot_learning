from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    average_precision_score
)
from EfficientNetB1Embedding import EfficientNetB1Embedding
from BuildSupportNQuery import *
from ProtoNet import *
from DataLoader import *

class StratifiedKFoldCrossValidation:
    def __init__(self,     
                model_name,
                dataset_path,
                k_folds=5,
                num_epochs=20,
                freeze_epochs=5,
                batch_size=32,
                lr=0.0001,
                weight_decay=1e-4,
                dropout_rate=0.3,
                focal_gamma=2.0,
                label_smoothing=0.1,
                use_class_aware_aug=True,
                use_weighted_sampling=True,
                use_temperature_scaling=True,
                calculate_cluster_metrics_flag=False,
                random_seed=42,
                pixel_weight=torch.ones((1, 1, 224, 224), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))):        
        # Store parameters
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.k_folds = k_folds
        self.num_epochs = num_epochs
        self.freeze_epochs = freeze_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.use_class_aware_aug = use_class_aware_aug
        self.use_weighted_sampling = use_weighted_sampling
        self.use_temperature_scaling = use_temperature_scaling
        self.calculate_cluster_metrics_flag = calculate_cluster_metrics_flag
        self.random_seed = random_seed
        self.pixel_weight = pixel_weight
        
        # Set seeds
        self._set_seed(random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self):
        #load dataset
        image_paths, labels, class_names, num_classes = load_dataset_from_folder(self.dataset_path) 
        image_paths = np.array(image_paths) 
        labels = np.array(labels)
        class_counts = np.bincount(labels)

        random_seed = self.random_seed
        skf = StratifiedKFold(
            n_splits=self.k_folds,
            shuffle=True,
            random_state=self.random_seed
        )
        
        fold_results = []
        all_preds = []
        all_labels = []
        fold_model = []
        ensemble_metrics = []
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "val_f1": [],
            "train_auc": [],
            "val_auc": [],
            "val_sens": []
        }



        for fold, (train_ids, val_ids) in enumerate(skf.split(image_paths, labels)):

            print(f"Fold {fold+1}")

            train_paths = image_paths[train_ids]
            train_labels = labels[train_ids]

            val_paths = image_paths[val_ids]
            val_labels = labels[val_ids]
            train_episode_builder = FewShotEpisodeBuilder(
                image_paths=train_paths,
                labels=train_labels,
                transform=train_transform,
                n_way=5,
                k_shot=1,
                q_query=5
            )

            val_episode_builder = FewShotEpisodeBuilder(
                image_paths=val_paths,
                labels=val_labels,
                transform=test_transform,
                n_way=5,
                k_shot=1,
                q_query=5
            )

            encoder = EfficientNetB1Embedding(pretrained=True)
            model = ProtoNet(encoder).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            fold_history = []

            for epoch in range(self.num_epochs):

                train_metrics = train_fewshot_epoch(
                    model,
                    train_episode_builder,
                    optimizer,
                    self.device,
                    episodes=200
                )

                val_metrics = validate_fewshot(
                    model,
                    val_episode_builder,
                    self.device,
                    num_episodes=100
                )

                epoch_result = {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_metrics": val_metrics
                }

                fold_history.append(epoch_result)

                print(f"\nEpoch {epoch+1}")
                print(val_metrics)


                
            fold_results.append({
                "fold": fold + 1,
                "history": fold_history,
                "best_val_acc": max([e["val_acc"] for e in fold_history])
            })
        return fold_history

def train_fewshot_epoch(model,
                        episode_builder,
                        optimizer,
                        device,
                        episodes=100):

    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    total_acc = 0

    for e in range(episodes):

        support_x, support_y, query_x, query_y = \
            episode_builder.sample_episode()

        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        logits = model(support_x, support_y, query_x)

        loss = criterion(logits, query_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        acc = (preds == query_y).float().mean()
        total_acc += acc.item()
        
        # print(f"Epoch {e+1}/{episodes} | "
        #     f"Train Loss: {total_loss/episodes:.4f} | "
        #     f"ACC: {total_acc/episodes:.4f}")

    return {
        "loss": total_loss / episodes,
        "accuracy": total_acc / episodes
    }


def validate_fewshot(model,
                    episode_builder,
                    device,
                    num_episodes=50):

    model.eval()

    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    with torch.no_grad():
        for _ in range(num_episodes):

            support_imgs, support_labels, query_imgs, query_labels = \
                episode_builder.sample_episode()

            support_imgs = support_imgs.to(device)
            support_labels = support_labels.to(device)   # FIX
            query_imgs = query_imgs.to(device)
            query_labels = query_labels.to(device)

            support_embeddings = model.extract_features(support_imgs)
            query_embeddings = model.extract_features(query_imgs)

            prototypes = []
            for c in torch.unique(support_labels):
                prototypes.append(
                    support_embeddings[support_labels == c].mean(0)
                )
            prototypes = torch.stack(prototypes)

            distances = torch.cdist(query_embeddings, prototypes)
            preds = torch.argmin(distances, dim=1)

            y_true = query_labels.cpu().numpy()
            y_pred = preds.cpu().numpy()

            acc = (preds == query_labels).float().mean().item()
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

    return {
        "accuracy_mean": float(np.mean(acc_list)),
        "accuracy_std": float(np.std(acc_list)),
        "precision_macro": float(np.mean(precision_list)),
        "recall_macro": float(np.mean(recall_list)),
        "f1_macro": float(np.mean(f1_list)),
        "episodes": num_episodes
    }
