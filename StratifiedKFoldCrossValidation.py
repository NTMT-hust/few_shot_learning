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
            best_val_acc = 0
            best_fold_model = None

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

                history["train_loss"].append(train_metrics["loss"])
                history["train_acc"].append(train_metrics["accuracy"])
                history["train_f1"].append(train_metrics["f1"])
                history["train_auc"].append(train_metrics["auc"])

                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])
                history["val_f1"].append(val_metrics["f1"])
                history["val_auc"].append(val_metrics["auc"])
                history["val_sens"].append(val_metrics["sensitivity"])


                print(f"\nEpoch {epoch+1}")
                print(val_metrics)
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    best_fold_model = model

            torch.save(
                best_fold_model.state_dict(),
                f"best_model_fold_{fold+1}.pth"
            )

            print("✓ Saved best model for this fold")
                
            fold_results.append({
                "fold": fold + 1,
                "history": history,
                "best_val_acc": max(history["val_acc"])
            })
            fold_model.append(model)

        return fold_results,fold_model, class_names


def train_fewshot_epoch(model, episode_builder, optimizer, device, episodes=100):

    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for _ in range(episodes):

        support_x, support_y, query_x, query_y = episode_builder.sample_episode()

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

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(query_y.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except:
        auc = 0.0

    return {
        "loss": total_loss / episodes,
        "accuracy": acc,
        "f1": f1,
        "auc": auc
    }


def validate_fewshot(model, episode_builder, device, num_episodes=50):

    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for _ in range(num_episodes):

            support_x, support_y, query_x, query_y = episode_builder.sample_episode()

            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)

            logits = model(support_x, support_y, query_x)
            loss = criterion(logits, query_y)

            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(query_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = np.mean(np.diag(cm) / (cm.sum(axis=1) + 1e-8))

    return {
        "loss": total_loss / num_episodes,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "sensitivity": sensitivity
    }
