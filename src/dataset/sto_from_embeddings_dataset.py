import logging
import os.path
import argparse
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tools import collate_single_embedding_fn

logger = logging.getLogger(__name__)

d_type = np.float32


class StoDataset(Dataset):

    def __init__(self, sto_labels_json: Path, embedding_path: Path, subset: float = 1.0):
        """
        Stoichiometry classes with esm3 embeddings dataset
        :param sto_labels_json: the json file with the stoichiometry data
        :param embedding_path: the directory containing the esm3 embeddings (one file per PDB chain)
        :param subset: subset of the whole data in sto_labels_json file to use: if positive we will use the given fraction starting from the left, if negative the given fraction starting from the right
        """
        self.embedding = {}
        self.data_points = list()
        self.max_length = 0
        self.subset = subset
        self.embedding_path = embedding_path
        self.num_classes = 0
        self.load_sto_classes(sto_labels_json)
        self.load_embedding()

    def load_sto_classes(self, sto_labels_json: Path):
        with open(sto_labels_json) as f:
            data = json.loads(f.read())
            for key in data.keys():
                # given our dataset of homomers, there should always be an "A" asym_id
                file = os.path.join(self.embedding_path, key + ".A.pt")
                if not os.path.exists(file):
                    continue
                self.data_points.append([
                    key, data[key]
                ])
            logger.info(f"Total data points with ESM embeddings: {len(self.data_points)}, out of a total in json file of {len(data)}")
            if self.subset > 0:
                end_index = int (self.subset * len(self.data_points))
                self.data_points = self.data_points[:end_index]
                logger.info("Using only %.1f of list from left. End index: %d. Final list size: %d" %
                      (self.subset, end_index, len(self.data_points)))
            else:
                start_index = int((1.0+self.subset) * len(self.data_points))
                self.data_points = self.data_points[start_index:]
                logger.info("Using only %.1f of list from right. Start index: %d. Final list size: %d" %
                      (self.subset, start_index, len(self.data_points)))

            counts = self.get_class_counts()
            self.num_classes = max(counts.keys())
            for size in sorted(counts.keys()):
                logger.info("Count of homomers with size %d : %d" % (size, counts[size]))

    def get_class_counts(self) -> dict:
        """
        :return: a dictionary of class label (stoichiometry) to counts of the class in the entire dataset
        """
        counts = {}
        for size in self.get_classes():
            if size not in counts:
                counts[size] = 0
            counts[size] = counts[size] + 1
        return counts

    def get_classes(self) -> list:
        """
        :return: a list of all class labels for the entire dataset
        """
        return [data_point[1] for data_point in self.data_points]

    def weights(self) -> torch.Tensor:
        """
        :return: a tensor of length num_samples with weights per sample
        """
        class_counts = self.get_class_counts()
        class_sample_count = np.array([class_counts[c] if c in class_counts else 0 for c in range(1, self.num_classes + 1)])
        weight = [] # per class weights (length = num_classes)
        for count in class_sample_count:
            if count!=0:
                weight.append(1.0/count)
            else:
                weight.append(0)
        samples_weight = np.array([weight[t - 1] for t in self.get_classes()])  # weights per sample (length = num_samples)
        return torch.from_numpy(samples_weight).double()

    def load_embedding(self):
        for data_point in self.data_points:
            pdb_id = data_point[0]
            file = pdb_id + ".A.pt"
            curr_embedding = torch.load(f"{self.embedding_path}/{file}", map_location=torch.device("cpu"))
            self.embedding[pdb_id] = curr_embedding
            if self.embedding[pdb_id].shape[0] > self.max_length:
                self.max_length = self.embedding[pdb_id].shape[0]

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        embedding_for_datapoint = self.embedding[self.data_points[idx][0]]
        label_probs = torch.zeros(self.num_classes)
        label_probs[self.data_points[idx][1] - 1] = 1.0
        return embedding_for_datapoint, label_probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sto_labels_json_file', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    args = parser.parse_args()

    dataset = StoDataset(
        sto_labels_json=args.sto_labels_json_file,
        embedding_path=args.embedding_path,
        subset=1.0
    )
    weights = dataset.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=16,
        collate_fn=collate_single_embedding_fn
    )
    for (x, x_mask), z in dataloader:
        print(z)
