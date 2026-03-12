import os
import random

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

batchsz = 1
seed = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = "syn"
image_file = r"./data/syn_dataset.mat"
P, L, col, N = 3, 188, 20, 400
endmember_names = None
lambda_rec = 1
lambda_vca = 4.5
lambda_kl = 0.001
lambda_sad = 3
lambda_consistency = 0.1
lambda_sad_recon = 3
lr = 5e-4
z_dim = 4
epochs = 700


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        print(f"Warning: failed to enable deterministic algorithms: {exc}")


def loadhsi(path):
    data = sio.loadmat(path)
    Y = torch.from_numpy(data["Y"].astype(np.float32)).reshape(L, col, col).unsqueeze(0)
    A_true = data["A"].astype(np.float32)
    A_true = A_true / np.sum(A_true, axis=0, keepdims=True)
    GT_M = data["M"].astype(np.float32)
    GT_M = GT_M / np.max(GT_M, axis=0, keepdims=True)
    VCA_M = data.get("M1")
    if VCA_M is not None:
        VCA_M = VCA_M.astype(np.float32)
    return Y, A_true, GT_M, VCA_M


set_seed(seed)
Y, A_true, GT_M, VCA_M = loadhsi(image_file)
train_db = torch.utils.data.DataLoader(Y, batch_size=batchsz, shuffle=True)

if VCA_M is not None:
    EM = torch.tensor(VCA_M.T.reshape(1, P, L)).to(device)
    use_vca = True
else:
    EM = None
    use_vca = False