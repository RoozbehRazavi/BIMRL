import torch
import numpy as np
import random
import sys
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Kmeans():

    def __init__(self, args):
        self.args = args
        self.dataset = torch.zeros(size=(self.args.kmeans_buffer_size, 2*self.args.latent_dim), device=device)
        self.num_centers = self.args.kmeans_num_cluster
        self.num_task = 0
        self.centers = self.random_init()

    def random_init(self):
        num_points = self.dataset.size(0)
        dimension = self.dataset.size(1)
        used = torch.zeros(num_points, dtype=torch.long, device=device)
        indices = torch.zeros(self.num_centers, dtype=torch.long, device=device)
        for i in range(self.num_centers):
            while True:
                cur_id = random.randint(0, num_points - 1)
                if used[cur_id] > 0:
                    continue
                used[cur_id] = 1
                indices[i] = cur_id
                break
        indices = indices.to(device)
        centers = torch.gather(self.dataset, 0, indices.view(-1, 1).expand(-1, dimension))
        return centers

    def compute_codes(self, dataset):
        num_points = dataset.size(0)
        dimension = dataset.size(1)
        num_centers = self.centers.size(0)
        chunk_size = int(5e8 / num_centers)
        codes = torch.zeros(num_points, dtype=torch.long, device=device)
        centers_t = torch.transpose(self.centers, 0, 1)
        centers_norms = torch.sum(self.centers ** 2, dim=1).view(1, -1)

        for i in range(0, num_points, chunk_size):
            begin = i
            end = min(begin + chunk_size, num_points)
            dataset_piece = dataset[begin:end, :]
            dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
            distances = torch.mm(dataset_piece, centers_t)
            distances *= -2.0
            distances += dataset_norms
            distances += centers_norms
            _, min_ind = torch.min(distances, dim=1)
            codes[begin:end] = min_ind
        return codes

    def update_centers(self, codes):
        num_points = self.dataset.size(0)
        dimension = self.dataset.size(1)
        centers = torch.zeros(self.num_centers, dimension, dtype=torch.float, device=device)
        cnt = torch.zeros(self.num_centers, dtype=torch.float, device=device)
        centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), self.dataset)
        cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=device))
        cnt = torch.where(cnt > 0.5, cnt, torch.ones(self.num_centers, dtype=torch.float, device=device))
        centers /= cnt.view(-1, 1)
        return centers

    def cluster(self):
        self.centers = self.random_init()
        codes = self.compute_codes(self.dataset)
        num_iterations = 0
        while True:
            sys.stdout.write('.')
            sys.stdout.flush()
            num_iterations += 1
            self.centers = self.update_centers(codes)
            new_codes = self.compute_codes(self.dataset)

            if torch.equal(codes, new_codes):
                break

            codes = new_codes
        return self.centers, codes

    def add_point(self, new_point):
        self.dataset = torch.cat((self.dataset, new_point))
        self.random_init()
        self.cluster()
        self.num_task += new_point.shape[0]

    def get_cluster(self, dataset):
        if self.num_task > 0:
            codes = self.compute_codes(dataset=dataset)
        else:
            codes = None

        return codes


