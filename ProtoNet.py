import torch
import torch.nn as nn
import timm

class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, support_x, support_y, query_x):
        # Encode
        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)

        n_way = len(torch.unique(support_y))

        # Compute prototypes
        prototypes = []
        for c in range(n_way):
            class_embeddings = support_embeddings[support_y == c]
            prototype = class_embeddings.mean(0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)  # (N, D)

        # Compute distance
        dists = torch.cdist(query_embeddings, prototypes)

        # Negative distance for softmax
        logits = -dists
        return logits
