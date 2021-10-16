import torch.nn as nn
import torch
from torch import optim
from memory.cvae import cVAE
from memory.ewc import EWC
from memory.sequentional_kmeans import Kmeans
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GenerativeDataset(Dataset):
    def __init__(self, dataset_key, dataset_value):
        super(GenerativeDataset, self).__init__()
        self.state_embedding = dataset_key
        self.brim_hidden_state = dataset_value

    def __getitem__(self, item):
        x1 = self.brim_hidden_state[item]
        x2 = self.state_embedding[item]
        return (x1, x2)

    def __len__(self):
        return len(self.brim_hidden_state)


def create_train_loader(train_dataset_key, train_dataset_value):
    dataset = GenerativeDataset(dataset_key=train_dataset_key, dataset_value=train_dataset_value)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)
    return dataloader


class Generative(object):
    def __init__(self, args, num_head):
        super(Generative, self).__init__()
        self.args = args
        self.old_task_size_thr = 1000
        self.num_head = num_head
        self.gene_models = nn.ModuleList([])
        self.optimizers = []
        self.importance = 0.5
        self.old_task_sample_size = 128
        self.lr = 0.00001
        self.kmeans = Kmeans(self.args)

        for i in range(3):
            gen_model = cVAE(obs_dim=self.args.state_embedding_size, brim_hidden_dim=2*self.args.brim_hidden_size[0]).to(device)
            self.gene_models.append(gen_model)
            self.optimizers.append(optim.Adam(params=gen_model.parameters(), lr=self.lr))

        self.old_task_sample = [[[], []], [[], []], [[], []]]

    def write(self, train_data, task_inference_id):
        self.kmeans.add_point(task_inference_id)
        clusters = self.kmeans.get_cluster(task_inference_id)
        cvae0_idx = ((clusters == 0).nonzero(as_tuple=True)[0])
        cvae1_idx = ((clusters == 1).nonzero(as_tuple=True)[0])
        cvae2_idx = ((clusters == 2).nonzero(as_tuple=True)[0])
        cva_dataset = []

        if not (cvae0_idx.shape[0] == 0):
            tmp1 = torch.cat([train_data[0][i] for i in cvae0_idx])
            tmp2 = torch.cat([train_data[1][i] for i in cvae0_idx])
            cva_dataset.append(create_train_loader(tmp1, tmp2))
        else:
            cva_dataset.append(None)

        if not (cvae1_idx.shape[0] == 0):
            tmp3 = torch.cat([train_data[0][i] for i in cvae1_idx])
            tmp4 = torch.cat([train_data[1][i] for i in cvae1_idx])
            cva_dataset.append(create_train_loader(tmp3, tmp4))
        else:
            cva_dataset.append(None)

        if not (cvae2_idx.shape[0] == 0):
            tmp5 = torch.cat([train_data[0][i] for i in cvae2_idx])
            tmp6 = torch.cat([train_data[1][i] for i in cvae2_idx])
            cva_dataset.append(create_train_loader(tmp5, tmp6))
        else:
            cva_dataset.append(None)

        for j in range(len(self.gene_models)):
            if (cva_dataset[j] is not None) and self.gene_models[j].training:
                if len(self.old_task_sample[j][0]) == 0:
                    for _ in range(self.args.num_epoch_for_gen_memory):
                        EWC.normal_train(self.gene_models[j], self.optimizers[j], cva_dataset[j])
                else:
                    old_tasks_dataset = self.get_sample_data_from_old_task(self.old_task_sample[j])
                    for _ in range(self.args.num_epoch_for_gen_memory):
                            EWC.ewc_train(self.gene_models[j], self.optimizers[j], cva_dataset[j], EWC(self.gene_models[j], old_tasks_dataset), self.importance)

        if len(self.old_task_sample[0][0]) > self.old_task_size_thr:
            self.old_task_sample[0][0] = []
            self.old_task_sample[0][1] = []

        if len(self.old_task_sample[1][0]) > self.old_task_size_thr:
            self.old_task_sample[1][0] = []
            self.old_task_sample[1][1] = []

        if len(self.old_task_sample[2][0]) > self.old_task_size_thr:
            self.old_task_sample[2][0] = []
            self.old_task_sample[2][1] = []

        if self.kmeans.dataset.shape[0] > self.old_task_size_thr:
            self.kmeans = Kmeans(self.args)

        if not (cvae0_idx.shape[0] == 0):
            self.old_task_sample[0][0].extend(list(tmp1))
            self.old_task_sample[0][1].extend(list(tmp2))

        if not (cvae1_idx.shape[0] == 0):
            self.old_task_sample[1][0].extend(list(tmp3))
            self.old_task_sample[1][1].extend(list(tmp4))

        if not (cvae2_idx.shape[0] == 0):
            self.old_task_sample[2][0].extend(list(tmp5))
            self.old_task_sample[2][1].extend(list(tmp6))

    def get_sample_data_from_old_task(self, dataset):
        idx = torch.randint(0, len(dataset[0]), size=(self.old_task_sample_size,)).long()
        brim_old_task = torch.stack(dataset[0])[idx]
        state_old_task = torch.stack(dataset[1])[idx]
        dataloader = create_train_loader(brim_old_task, state_old_task)
        return dataloader

    def read(self, task_id, obs_embdd):
        cluster_idx = self.kmeans.get_cluster(task_id)
        if cluster_idx == None:
            return torch.zeros(size=(obs_embdd.shape[0], self.num_head*2*self.args.brim_hidden_size[0]), device=device)

        cvae0_idx = ((cluster_idx == 0).nonzero(as_tuple=True)[0])
        cvae1_idx = ((cluster_idx == 1).nonzero(as_tuple=True)[0])
        cvae2_idx = ((cluster_idx == 2).nonzero(as_tuple=True)[0])

        result = torch.zeros(size=(obs_embdd.shape[0], self.num_head * 2*self.args.brim_hidden_size[0]), device=device)
        data = list()
        data.append(obs_embdd[cvae0_idx])
        data.append(obs_embdd[cvae1_idx])
        data.append(obs_embdd[cvae2_idx])

        for i in range(len(self.gene_models)):
            tmp = self.gene_models[i].generate(data[i].view(-1, self.args.state_embedding_size)).to(device)
            tmp = tmp.reshape(len(data[i]), self.num_head * 2 * self.args.brim_hidden_size[0])
            #tmp = linear_output(tmp)

            if i == 0:
                result[cvae0_idx] = tmp
            elif i == 1:
                result[cvae1_idx] = tmp
            elif i ==2:
                result[cvae2_idx] = tmp
        return result.clone().detach()
