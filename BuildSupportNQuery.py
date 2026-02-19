from collections import defaultdict
import random
import torch
from PIL import Image
from torchvision import transforms

import numpy as np
import random
from collections import defaultdict
from PIL import Image

class FewShotEpisodeBuilder:
    def __init__(self, image_paths, labels, transform,
                n_way=5, k_shot=1, q_query=5):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())

    def load_image(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def sample_episode(self):
        selected_classes = random.sample(self.classes, self.n_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for new_label, cls in enumerate(selected_classes):

            indices = random.choices(
                self.class_to_indices[cls],
                k=self.k_shot + self.q_query
            )

            support_idx = indices[:self.k_shot]
            query_idx = indices[self.k_shot:]

            for idx in support_idx:
                support_x.append(self.load_image(idx))
                support_y.append(new_label)

            for idx in query_idx:
                query_x.append(self.load_image(idx))
                query_y.append(new_label)

        support_x = torch.stack(support_x)
        query_x = torch.stack(query_x)

        support_y = torch.tensor(support_y)
        query_y = torch.tensor(query_y)

        return support_x, support_y, query_x, query_y


def build_class_indices(labels):
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)
    return class_to_indices
train_transform = transforms.Compose([ transforms.Resize((240, 240)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
test_transform = train_transform
def load_and_transform(path):
    img = Image.open(path).convert("RGB")
    img = train_transform(img)   # your torchvision transform
    return img
