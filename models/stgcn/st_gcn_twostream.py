import torch
import torch.nn as nn

from models.stgcn.st_gcn import Model as ST_GCN

class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)

    def forward(self, x):
        if len(x.size()) == 4:
            x = x.unsqueeze(4)


        N, C, T, V, M = x.size()
        m = torch.cat((torch.FloatTensor(N, C, 1, V, M).zero_().to(x.device),
                        x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
                        torch.FloatTensor(N, C, 1, V, M).zero_().to(x.device)), 2)

        res = self.origin_stream(x) + self.motion_stream(m)
        return res


if __name__ == '__main__':
    model = Model().cuda(0)
    data = torch.ones((2, 3, 12, 18))
    data = data.to(0)
    out = model(data)
    print(out)