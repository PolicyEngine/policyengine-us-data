import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HardConcrete(nn.Module):
    """HardConcrete distribution for L0 regularization."""

    def __init__(
        self,
        input_dim,
        output_dim=None,
        temperature=0.5,
        stretch=0.1,
        init_mean=0.5,
    ):
        super().__init__()
        if output_dim is None:
            self.gate_size = (input_dim,)
        else:
            self.gate_size = (input_dim, output_dim)
        self.qz_logits = nn.Parameter(torch.zeros(self.gate_size))
        self.temperature = temperature
        self.stretch = stretch
        self.gamma = -0.1
        self.zeta = 1.1
        self.init_mean = init_mean
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_mean is not None:
            init_val = math.log(self.init_mean / (1 - self.init_mean))
            self.qz_logits.data.fill_(init_val)

    def forward(self, input_shape=None):
        if self.training:
            gates = self._sample_gates()
        else:
            gates = self._deterministic_gates()
        if input_shape is not None and len(input_shape) > len(gates.shape):
            gates = gates.unsqueeze(-1).unsqueeze(-1)
        return gates

    def _sample_gates(self):
        u = torch.zeros_like(self.qz_logits).uniform_(1e-8, 1.0 - 1e-8)
        s = torch.log(u) - torch.log(1 - u) + self.qz_logits
        s = torch.sigmoid(s / self.temperature)
        s = s * (self.zeta - self.gamma) + self.gamma
        gates = torch.clamp(s, 0, 1)
        return gates

    def _deterministic_gates(self):
        probs = torch.sigmoid(self.qz_logits)
        gates = probs * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(gates, 0, 1)

    def get_penalty(self):
        logits_shifted = self.qz_logits - self.temperature * math.log(
            -self.gamma / self.zeta
        )
        prob_active = torch.sigmoid(logits_shifted)
        return prob_active.sum()

    def get_active_prob(self):
        logits_shifted = self.qz_logits - self.temperature * math.log(
            -self.gamma / self.zeta
        )
        return torch.sigmoid(logits_shifted)


class L0Linear(nn.Module):
    """Linear layer with L0 regularization using HardConcrete gates."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        temperature=0.5,
        init_sparsity=0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.weight_gates = HardConcrete(
            out_features,
            in_features,
            temperature=temperature,
            init_mean=init_sparsity,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode="fan_out")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        gates = self.weight_gates()
        masked_weight = self.weight * gates
        return F.linear(input, masked_weight, self.bias)

    def get_l0_penalty(self):
        return self.weight_gates.get_penalty()

    def get_sparsity(self):
        with torch.no_grad():
            prob_active = self.weight_gates.get_active_prob()
            return 1.0 - prob_active.mean().item()


class SparseMLP(nn.Module):
    """Example MLP with L0 regularization on all layers"""

    def __init__(
        self,
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
        init_sparsity=0.5,
        temperature=0.5,
    ):
        super().__init__()
        self.fc1 = L0Linear(
            input_dim,
            hidden_dim,
            init_sparsity=init_sparsity,
            temperature=temperature,
        )
        self.fc2 = L0Linear(
            hidden_dim,
            hidden_dim,
            init_sparsity=init_sparsity,
            temperature=temperature,
        )
        self.fc3 = L0Linear(
            hidden_dim,
            output_dim,
            init_sparsity=init_sparsity,
            temperature=temperature,
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_l0_loss(self):
        l0_loss = 0
        for module in self.modules():
            if isinstance(module, L0Linear):
                l0_loss += module.get_l0_penalty()
        return l0_loss

    def get_sparsity_stats(self):
        stats = {}
        for name, module in self.named_modules():
            if isinstance(module, L0Linear):
                stats[name] = {
                    "sparsity": module.get_sparsity(),
                    "active_params": module.get_l0_penalty().item(),
                }
        return stats


def train_with_l0(model, train_loader, epochs=10, l0_lambda=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        total_l0 = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            ce_loss = criterion(output, target)
            l0_loss = model.get_l0_loss()
            loss = ce_loss + l0_lambda * l0_loss
            loss.backward()
            optimizer.step()
            total_loss += ce_loss.item()
            total_l0 += l0_loss.item()
        if epoch % 1 == 0:
            sparsity_stats = model.get_sparsity_stats()
            print(
                f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, L0={total_l0/len(train_loader):.4f}"
            )
            for layer, stats in sparsity_stats.items():
                print(
                    f"  {layer}: {stats['sparsity']*100:.1f}% sparse, {stats['active_params']:.1f} active params"
                )


def prune_model(model, threshold=0.05):
    for module in model.modules():
        if isinstance(module, L0Linear):
            with torch.no_grad():
                prob_active = module.weight_gates.get_active_prob()
                mask = (prob_active > threshold).float()
                module.weight.data *= mask
    return model
