import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = int((kernel_size - 1) / 2)
        self.Wxi = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whi = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxf = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whf = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxc = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whc = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxo = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Who = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

        self.prev_hidden = None

    def forward(self, x):
        h, c = self.prev_hidden
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        self.prev_hidden = ch, cc

        return ch, cc

    def reset(self, done):
        h = self.prev_hidden[0] * (1 - done.unsqueeze(-1).unsqueeze(-1))
        c = self.prev_hidden[1] * (1 - done.unsqueeze(-1).unsqueeze(-1))
        self.prev_hidden = (h, c)

    def prior(self, batch_size):
        self.prev_hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        height = 5
        width = 5
        if self.Wci is None:
            self.Wci = torch.zeros(1, self.hidden_channels, height, width, requires_grad=True).to(
                device
            )
            self.Wcf = torch.zeros(1, self.hidden_channels, height, width, requires_grad=True).to(
                device
            )
            self.Wco = torch.zeros(1, self.hidden_channels, height, width, requires_grad=True).to(
                device
            )
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, requires_grad=True).to(
                device
            ),
            torch.zeros(batch_size, self.hidden_channels, height, width, requires_grad=True).to(
                device
            ),
        )


class VisionNetwork(nn.Module):
    def __init__(self):
        super(VisionNetwork, self).__init__()
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=(5, 5),
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=8,
                out_channels=12,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
        )
        self.vision_lstm = ConvLSTMCell(
            input_channels=8, hidden_channels=12, kernel_size=3
        )

    def reset(self, done):
        self.vision_lstm.reset(done)

    def prior(self, batch_size):
        self.vision_lstm.prior(batch_size)

    def forward(self, X):
        X = X.transpose(1, 3)
        tmp = self.vision_cnn(X)
        #O, _ = self.vision_lstm(tmp)
        return tmp.transpose(1, 3)


class QueryNetwork(nn.Module):
    def __init__(self, hidden_state_size):
        super(QueryNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def forward(self, query):
        out = self.model(query)
        return out.reshape(-1, 4, 8)


class SpatialBasis:
    """
    NOTE: The `height` and `weight` depend on the inputs' size and its resulting size
    after being processed by the vision network.
    """

    def __init__(self, height=5, width=5, channels=4):
        h, w, d = height, width, channels

        p_h = torch.mul(torch.arange(1, h+1).unsqueeze(1).float(), torch.ones(1, w).float()) * (np.pi / h)
        p_w = torch.mul(torch.ones(h, 1).float(), torch.arange(1, w+1).unsqueeze(0).float()) * (np.pi / w)
        
        # NOTE: I didn't quite see how U,V = 4 made sense given that the authors form the spatial
        # basis by taking the outer product of the values. Still, I think what I have is aligned with what
        # they did, but I am less confident in this step.
        U = V = 2 # size of U, V.
        u_basis = v_basis = torch.arange(1, U+1).unsqueeze(0).float()
        a = torch.mul(p_h.unsqueeze(2), u_basis)
        b = torch.mul(p_w.unsqueeze(2), v_basis)
        out = torch.einsum('hwu,hwv->hwuv', torch.cos(a), torch.cos(b))
        out = out.reshape(h, w, d)
        self.S = out

    def __call__(self, X):
        # Stack the spatial bias (for each batch) and concat to the input.
        batch_size = X.size()[0]
        S = torch.stack([self.S] * batch_size).to(X.device)
        return torch.cat([X, S], dim=3)


def spatial_softmax(A):
    # A: batch_size x h x w x d
    b, h, w, d = A.size()
    # Flatten A s.t. softmax is applied to each grid (not over queries)
    A = A.reshape(b, h * w, d)
    A = F.softmax(A, dim=1)
    # Reshape A to original shape.
    A = A.reshape(b, h, w, d)
    return A


def apply_alpha(A, V):
    # TODO: Check this function again.
    b, h, w, c = A.size()
    A = A.reshape(b, h * w, c).transpose(1, 2)

    _, _, _, d = V.size()
    V = V.reshape(b, h * w, d)

    return torch.matmul(A, V)


class VisionCore(nn.Module):
    def __init__(
        self,
        hidden_size,
        c_v,
        c_k,
        c_s,
        num_queries,
        state_embedding_size,
    ):
        """Agent implementing the attention agent.
        """
        super(VisionCore, self).__init__()
        self.hidden_size = hidden_size
        self.c_v, self.c_k, self.c_s, self.num_queries = c_v, c_k, c_s, num_queries

        self.vision = VisionNetwork()
        self.query = QueryNetwork(hidden_state_size=hidden_size)
        # TODO: Implement SpatialBasis.
        self.spatial = SpatialBasis()

        self.answer_processor = nn.Sequential(
            # 1026 x 512
            nn.Linear(
                (c_v + c_s) * num_queries, 32
            ),
            nn.ReLU(),
            nn.Linear(32, state_embedding_size-1),
        )

    def reset(self, done):
        self.vision.reset(done)

    def prior(self, batch_size):
        self.vision.prior(batch_size)

    def detach_hidden_state(self):
        self.vision.vision_lstm.prev_hidden = (
        self.vision.vision_lstm.prev_hidden[0].detach(), self.vision.vision_lstm.prev_hidden[1].detach())

    def forward(self, X, upper_module_hidden_state):

        # 0. Setup.
        # ---------
        batch_size = X.size()[0]
        # 1 (a). Vision.
        # --------------

        # (n, h, w, c_k + c_v)
        O = self.vision(X)
        # (n, h, w, c_k), (n, h, w, c_v)
        K, V = O.split([self.c_k, self.c_v], dim=3)
        # (n, h, w, c_k + c_s), (n, h, w, c_v + c_s)
        K, V = self.spatial(K), self.spatial(V)

        # 1 (b). Queries.
        # --------------
        # (n, h, w, num_queries, c_k + c_s)
        Q = self.query(upper_module_hidden_state)

        # 2. Answer.
        # ----------
        # (n, h, w, num_queries)
        A = torch.matmul(K, Q.transpose(2, 1).unsqueeze(1))
        # (n, h, w, num_queries)
        A = spatial_softmax(A)
        # (n, 1, 1, num_queries)
        a = apply_alpha(A, V)

        # (n, (c_v + c_s) * num_queries + (c_k + c_s) * num_queries + 1 + 1)
        a = torch.chunk(a, self.num_queries, dim=1)
        answer = torch.cat(a, dim=2).squeeze(1)
        # (n, hidden_size)
        answer = self.answer_processor(answer)

        return answer
