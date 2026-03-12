import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import P, col


def conv_spatial_consistency(a, col, P, device, kernel_type="laplace"):
    a = a.contiguous().to(device)
    a_img = a.view(col, col, P).permute(2, 0, 1).unsqueeze(0)

    if kernel_type == "laplace":
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
    else:
        kernel = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

    weight = kernel.unsqueeze(0).unsqueeze(0).repeat(P, 1, 1, 1)
    conv_out = F.conv2d(a_img, weight, stride=1, padding=1, groups=P)
    return (conv_out ** 2).mean()


def abundance_rmse(inputsrc, inputref):
    return np.sqrt(((inputsrc - inputref) ** 2).mean())


def sad_distance(src, ref):
    cos_sim = np.dot(src, ref) / ((np.linalg.norm(src) * np.linalg.norm(ref)) + 1e-8)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.arccos(cos_sim)


def evaluate(A_hat, A_true, EM_hat, EM_true):
    rmse_matrix = np.zeros((P, P))
    sad_matrix = np.zeros((P, P))
    rmse_a = np.zeros(P)
    sad_m = np.zeros(P)

    for i in range(P):
        for j in range(P):
            rmse_matrix[i, j] = abundance_rmse(A_hat[i, :, :], A_true[j, :, :])
            sad_matrix[i, j] = sad_distance(EM_hat[:, i], EM_true[:, j])
        rmse_a[i] = np.min(rmse_matrix[i, :])
        sad_m[i] = np.min(sad_matrix[i, :])

    return rmse_a, sad_m


def sort_MA(A_hat, A_true):
    dev = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            dev[i, j] = np.mean((A_hat[i, :] - A_true[j, :]) ** 2)
    return np.argmin(dev, axis=0)


def adjust_Adim(A_hat, A_true):
    return A_hat.reshape(P, col, col), A_true.reshape(P, col, col)


def plot_abundance(A_hat, A_true):
    A_true = A_true / np.clip(np.sum(A_true, axis=0, keepdims=True), 1e-8, None)
    A_hat = A_hat / np.clip(np.sum(A_hat, axis=0, keepdims=True), 1e-8, None)

    plt.figure(figsize=(3 * P, 6))
    for i in range(P):
        plt.subplot(2, P, i + 1)
        estimated = plt.imshow(A_hat[i], cmap="jet", interpolation="none")
        estimated.set_clim(vmin=0, vmax=1)
        plt.axis("off")
        plt.title(f"Est {i + 1}")

        plt.subplot(2, P, i + P + 1)
        truth = plt.imshow(A_true[i], cmap="jet", interpolation="none")
        truth.set_clim(vmin=0, vmax=1)
        plt.axis("off")
        plt.title(f"GT {i + 1}")

    plt.tight_layout()
    plt.show()


def plot_EM_compare(EM_hat_avg, EM_true, endmember_names=None):
    EM_hat_avg = np.asarray(EM_hat_avg)
    EM_true = np.asarray(EM_true)
    _, P_local = EM_hat_avg.shape
    cols = min(3, P_local)
    rows = int(np.ceil(P_local / cols))

    plt.figure(figsize=(5 * cols, 3 * rows))
    for i in range(P_local):
        plt.subplot(rows, cols, i + 1)
        plt.plot(EM_hat_avg[:, i], label="Estimated", color="royalblue")
        plt.plot(EM_true[:, i], label="GroundTruth", color="orange", linestyle="--")
        plt.xlabel("Bands")
        plt.ylabel("Reflectance")
        plt.ylim(0, 1)
        if endmember_names is None:
            plt.title(f"Endmember {i + 1}")
        else:
            plt.title(str(endmember_names[i]))
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.show()