import torch
import torch.nn as nn
from arguments import OptimizationParams


class TransferMLP:
    
    def __init__(self, sh_degree, features_n):
        self.sh_degree = sh_degree
        self.hidden_dim = 64
        self.coeffs_n = (self.sh_degree + 1) ** 2

        self.net = nn.ModuleList()
        self.net.append(nn.Linear(3, self.hidden_dim, bias=False, device="cuda"))
        self.net.append(nn.ReLU(False))
        self.net.append(nn.Linear(self.hidden_dim + features_n, self.hidden_dim, bias=False, device="cuda"))
        self.net.append(nn.ReLU(False))
        self.net.append(nn.Linear(self.hidden_dim, self.coeffs_n, bias=False, device="cuda"))

    def forward(self, features, dir): 
        x = self.net[0](dir)
        x = self.net[1](x)
        x = torch.cat((x, features), dim=1)
        x = self.net[2](x)
        x = self.net[3](x)
        x = self.net[4](x)
        return x

    def training_setup(self, training_args: OptimizationParams):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=training_args.mlp_lr, weight_decay=1e-6, eps=1e-15)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def capture(self):
        captured_list = [
            self.net,
            self.optimizer.state_dict(),
        ]

        return captured_list


    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.net,
         opt_dict) = model_args[:2]

        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter

