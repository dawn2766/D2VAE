import numpy as np
import torch

from data import A_true, EM, GT_M, L, P, Y, col, dataset, device, endmember_names, epochs, lambda_consistency, lambda_kl, lambda_rec, lambda_sad, lambda_sad_recon, lambda_vca, lr, train_db, use_vca, z_dim
from model import D2VAE
from utils import adjust_Adim, conv_spatial_consistency, evaluate, plot_EM_compare, plot_abundance, sort_MA

model = D2VAE(P, L, z_dim, col).to(device)
model.apply(model.weights_init)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5)


def train():
    for epoch in range(epochs):
        model.train()
        for y in train_db:
            y = y.to(device)
            y_hat, mu, log_var, a, em_tensor = model(y)

            y = y.permute(2, 3, 0, 1).reshape(col * col, L)
            loss_rec = ((y_hat - y) ** 2).sum() / y.shape[0]

            loss_kl = -0.5 * (log_var + 1 - mu ** 2 - log_var.exp())
            loss_kl = loss_kl.sum() / y.shape[0]
            loss_kl = torch.maximum(loss_kl, torch.tensor(0.2, device=device))

            loss_consistency = conv_spatial_consistency(a, col, P, device)

            dot_product = (y * y_hat).sum(dim=1)
            y_norm = y.square().sum(dim=1).sqrt()
            y_hat_norm = y_hat.square().sum(dim=1).sqrt()
            cos_similarity = dot_product / (y_norm * y_hat_norm + 1e-8)
            cos_similarity = torch.clamp(cos_similarity, -1.0 + 1e-7, 1.0 - 1e-7)
            loss_sad_recon = torch.acos(cos_similarity).mean()

            loss = lambda_rec * loss_rec + lambda_kl * loss_kl
            loss = loss + lambda_consistency * loss_consistency + lambda_sad_recon * loss_sad_recon

            if use_vca and EM is not None and epoch < epochs // 2:
                loss_vca = (em_tensor - EM).square().sum() / y.shape[0]
                loss = loss + lambda_vca * loss_vca
            else:
                em_bar = em_tensor.mean(dim=0, keepdim=True)
                aa = (em_tensor * em_bar).sum(dim=2)
                em_bar_norm = em_bar.square().sum(dim=2).sqrt()
                em_tensor_norm = em_tensor.square().sum(dim=2).sqrt()
                cos_val = aa / (em_bar_norm + 1e-6) / (em_tensor_norm + 1e-6)
                cos_val = torch.clamp(cos_val, -1.0 + 1e-7, 1.0 - 1e-7)
                loss_sad = torch.acos(cos_val).sum() / y.shape[0] / P
                loss = loss + lambda_sad * loss_sad

            optimizer.zero_grad()
            loss.backward()

            for param in model.parameters():
                if param.grad is not None:
                    invalid = torch.isnan(param.grad) | torch.isinf(param.grad)
                    if invalid.any():
                        param.grad = torch.where(invalid, torch.zeros_like(param.grad), param.grad)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"训练在第 {epoch + 1} 轮中断，loss={loss.item()}")
            return False

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch: {epoch + 1:03d}",
                f"| loss_rec: {loss_rec.item():.4f}",
                f"| loss_kl: {loss_kl.item():.4f}",
                f"| loss_consistency: {loss_consistency.item():.4f}",
                f"| loss_sad_recon: {loss_sad_recon.item():.4f}",
                f"| loss: {loss.item():.4f}",
            )

    return True


def evaluate_and_visualize():
    model.eval()
    with torch.no_grad():
        _, _, _, A, em_hat = model(Y.to(device))

    A_hat = A.cpu().numpy().T
    pos = sort_MA(A_hat, A_true)
    A_hat = A_hat[pos, :]
    em_hat = em_hat[:, pos, :].cpu().numpy()

    em_list = []
    for index in range(P):
        weights = A_hat[index, :]
        weight_sum = np.sum(weights)
        if weight_sum <= 1e-8:
            weighted_spec = em_hat[:, index, :].mean(axis=0)
        else:
            weighted_spec = np.sum(em_hat[:, index, :] * weights[:, None], axis=0) / weight_sum
        max_value = np.max(weighted_spec)
        if max_value > 1e-8:
            weighted_spec = weighted_spec / max_value
        em_list.append(weighted_spec)

    em_one = np.stack(em_list, axis=1)
    A_hat, A_true_adj = adjust_Adim(A_hat, A_true)
    rmse_a, sad_m = evaluate(A_hat, A_true_adj, em_one, GT_M)

    print("dataset", dataset)
    print("RMSE", rmse_a)
    print("mean_RMSE", rmse_a.mean())
    print("SAD", sad_m)
    print("mean_SAD", sad_m.mean())

    plot_abundance(A_hat, A_true_adj)
    plot_EM_compare(em_one, GT_M, endmember_names)


if __name__ == "__main__":
    train_success = train()
    if train_success:
        evaluate_and_visualize()