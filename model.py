import torch
import torch.nn as nn
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

class SVSRNN(nn.Module):
    def __init__(self, num_features, num_rnn_layer=4, num_hidden_units=[256, 256, 256], tensorboard_directory='graphs/svsrnn', clear_tensorboard=True):
        super(SVSRNN, self).__init__()

        assert len(num_hidden_units) == num_rnn_layer

        self.num_features = num_features
        self.num_rnn_layer = num_rnn_layer
        self.num_hidden_units = num_hidden_units

        self.gstep = 0

        # GRU Layers
        self.rnn_layers = nn.ModuleList([
            nn.GRU(input_size=num_features if i == 0 else num_hidden_units[i - 1],
                   hidden_size=units,
                   batch_first=True)
            for i, units in enumerate(self.num_hidden_units)
        ])

        # Projection layer
        self.projection = nn.Linear(num_hidden_units[-1], num_features)

        # Two output dense heads for source 1 and source 2
        self.dense_src1 = nn.Linear(num_features, num_features)
        self.dense_src2 = nn.Linear(num_features, num_features)

        self.gamma = 0.001
        self.optimizer = None

        if clear_tensorboard:
            shutil.rmtree(tensorboard_directory, ignore_errors=True)
        self.writer = SummaryWriter(tensorboard_directory)

    def forward(self, x, training=False):
        # x shape: [batch, time, features]
        out = x
        for rnn in self.rnn_layers:
            out, _ = rnn(out)

        # Project RNN output to feature dimension
        out = self.projection(out)

        y_hat_src1 = torch.relu(self.dense_src1(out))
        y_hat_src2 = torch.relu(self.dense_src2(out))

        mask_logits = torch.stack([y_hat_src1, y_hat_src2], dim=-1)  # [B, T, F, 2]
        mask = torch.softmax(mask_logits, dim=-1)

        y_tilde_src1 = mask[..., 0] * x
        y_tilde_src2 = mask[..., 1] * x

        return y_tilde_src1, y_tilde_src2

    def si_snr(self, target, estimate, eps=1e-8):
        target = target.reshape(target.shape[0], -1)
        estimate = estimate.reshape(estimate.shape[0], -1)
        target_mean = torch.mean(target, dim=1, keepdim=True)
        estimate_mean = torch.mean(estimate, dim=1, keepdim=True)

        target_zm = target - target_mean
        estimate_zm = estimate - estimate_mean

        s_target = torch.sum(estimate_zm * target_zm, dim=1, keepdim=True) * target_zm / (
            torch.sum(target_zm ** 2, dim=1, keepdim=True) + eps)
        e_noise = estimate_zm - s_target

        s_target_energy = torch.sum(s_target ** 2, dim=1) + eps
        e_noise_energy = torch.sum(e_noise ** 2, dim=1) + eps

        si_snr = 10 * torch.log10(s_target_energy / e_noise_energy)
        return si_snr

    def loss_fn(self, y_src1, y_src2, y_pred_src1, y_pred_src2):
        si_snr1 = self.si_snr(y_src1, y_pred_src1)
        si_snr2 = self.si_snr(y_src2, y_pred_src2)
        loss = -(torch.mean(si_snr1) + torch.mean(si_snr2)) / 2.0
        return loss

    def train_step(self, x, y1, y2):
        self.optimizer.zero_grad()
        y_pred_src1, y_pred_src2 = self(x, training=True)
        loss = self.loss_fn(y1, y2, y_pred_src1, y_pred_src2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.optimizer.step()
        return loss.item(), y_pred_src1, y_pred_src2

    def train(self, x, y1, y2, learning_rate):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

        loss, y_pred_src1, y_pred_src2 = self.train_step(x, y1, y2)
        self.summary(x, y1, y2, y_pred_src1, y_pred_src2, loss)
        return loss

    def validate(self, x, y1, y2):
        with torch.no_grad():
            y1_pred, y2_pred = self(x, training=False)
            validate_loss = self.loss_fn(y1, y2, y1_pred, y2_pred)
        return y1_pred, y2_pred, validate_loss.item()

    def test(self, x):
        with torch.no_grad():
            return self(x, training=False)

    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not filename.endswith('.pth'):
            filename = filename.rsplit('.', 1)[0] + '.pth'
        torch.save(self.state_dict(), os.path.join(directory, filename))
        return os.path.join(directory, filename)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))

    def summary(self, x, y1, y2, y_pred_src1, y_pred_src2, loss):
        self.writer.add_scalar('loss', loss, self.gstep)
        self.writer.add_histogram('x_mixed', x, self.gstep)
        self.writer.add_histogram('y_src1', y1, self.gstep)
        self.writer.add_histogram('y_src2', y2, self.gstep)
        self.writer.add_histogram('y_pred_src1', y_pred_src1, self.gstep)
        self.writer.add_histogram('y_pred_src2', y_pred_src2, self.gstep)
        self.writer.flush()
        self.gstep += 1
