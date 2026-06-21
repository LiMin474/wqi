import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import differential_evolution
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class RegMLP(nn.Module):
    def __init__(self, n_features, n_layers, layer1, layer2, activation, dropout_rate):
        super().__init__()
        act_map = {'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU()}
        act_fn = act_map[activation]

        if n_layers == 1:
            self.net = nn.Sequential(
                nn.Linear(n_features, layer1),
                act_fn,
                nn.Dropout(dropout_rate),
                nn.Linear(layer1, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(n_features, layer1),
                act_fn,
                nn.Dropout(dropout_rate),
                nn.Linear(layer1, layer2),
                act_fn,
                nn.Dropout(dropout_rate),
                nn.Linear(layer2, 1)
            )

    def forward(self, x):
        return self.net(x)


class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_reg_mlp(X_train, y_train, X_val, y_val, n_layers, layer1, layer2,
                  activation, alpha_l2, alpha_l1, dropout_rate, max_epochs=2000):
    n_features = X_train.shape[1]
    model = RegMLP(n_features, n_layers, layer1, layer2, activation, dropout_rate)

    scaler = StandardScaler()
    X_train_s = torch.FloatTensor(scaler.fit_transform(X_train))
    y_train_s = torch.FloatTensor(y_train.values.ravel() if hasattr(y_train, 'values') else y_train.ravel()).view(-1, 1)

    if X_val is not None:
        X_val_s = torch.FloatTensor(scaler.transform(X_val))
        y_val_s = torch.FloatTensor(y_val.values.ravel() if hasattr(y_val, 'values') else y_val.ravel()).view(-1, 1)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=10,
        tolerance_change=1e-9,
        tolerance_grad=1e-7
    )

    stopper = EarlyStopping(patience=30)
    best_val_loss = float('inf')
    best_state = None
    n_epochs_run = 0

    for epoch in range(max_epochs):
        n_epochs_run = epoch + 1

        def closure():
            optimizer.zero_grad()
            output = model(X_train_s)

            mse = F.mse_loss(output, y_train_s)

            l1_penalty = sum(p.abs().sum() for p in model.parameters())

            loss = mse + alpha_l1 * l1_penalty
            loss.backward()
            return loss

        optimizer.step(closure)

        if X_val is not None:
            with torch.no_grad():
                val_output = model(X_val_s)
                val_loss = F.mse_loss(val_output, y_val_s).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

                if stopper.step(val_loss):
                    n_epochs_run = epoch + 1
                    break
        else:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        pred_train = model(X_train_s).detach().cpu().numpy().ravel()

    return model, scaler, pred_train, n_epochs_run


def decode_params(x):
    n_layers = 1 if x[0] < 0.5 else 2
    layer1 = int(round(2 + x[1] * 30))
    layer1 = max(2, min(32, layer1))
    layer2 = int(round(2 + x[2] * 30))
    layer2 = max(2, min(32, layer2))
    act_idx = min(int(x[3] * 3), 2)
    activation = ['tanh', 'sigmoid', 'relu'][act_idx]
    alpha_l2 = 10.0 ** (-6.0 + x[4] * 5.0)
    alpha_l1 = 10.0 ** (-6.0 + x[5] * 5.0)
    dropout_rate = x[6] * 0.5
    return n_layers, layer1, layer2, activation, alpha_l2, alpha_l1, dropout_rate


def SumSqr_Reg(params, XX, YY, cvss):
    n_layers, layer1, layer2, activation, alpha_l2, alpha_l1, dropout_rate = params

    y_all = YY.values.ravel() if hasattr(YY, 'values') else YY.ravel()

    SST = np.sum((y_all - np.mean(y_all))**2)

    all_preds = np.zeros_like(y_all)

    for train_idx, val_idx in cvss:
        X_tr, X_va = XX[train_idx], XX[val_idx]
        y_tr, y_va = y_all[train_idx], y_all[val_idx]

        model, scaler, pred_train, n_epochs = train_reg_mlp(
            X_tr, y_tr, X_va, y_va,
            n_layers, layer1, layer2, activation,
            alpha_l2, alpha_l1, dropout_rate
        )

        with torch.no_grad():
            X_va_s = torch.FloatTensor(scaler.transform(X_va))
            val_pred = model(X_va_s).detach().cpu().numpy().ravel()

        all_preds[val_idx] = val_pred

    SSEcv = np.sum((y_all - all_preds)**2)
    R2CV = 1 - (SSEcv / SST)

    model_full, scaler_full, pred_full, n_epochs_full = train_reg_mlp(
        XX, y_all, None, None,
        n_layers, layer1, layer2, activation,
        alpha_l2, alpha_l1, dropout_rate
    )

    SSEmdl = np.sum((y_all - pred_full)**2)
    R2 = 1 - (SSEmdl / SST)

    output = {'R2': R2, 'R2CV': R2CV, 'Mdl': (model_full, scaler_full)}
    target = 1 - R2CV
    return target, output


def a4_DE_fitrnet_opt_reg(Pred, Resp):
    numFolds = 5
    np.random.seed(1)

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=1)
    cvss = list(kf.split(Pred))

    n_params = 7
    bounds = [(0.0, 1.0)] * n_params

    eval_count = [0]
    best_target = [float('inf')]
    best_r2cv = [0.0]
    convergence_history = []

    def objective(x):
        params = decode_params(x)
        target, output = SumSqr_Reg(params, Pred, Resp, cvss)
        eval_count[0] += 1
        if target < best_target[0]:
            best_target[0] = target
            best_r2cv[0] = output['R2CV']
            convergence_history.append((eval_count[0], best_r2cv[0]))
            print(f'  DE-Reg eval {eval_count[0]:3d}: best so far -> '
                  f'R2={output["R2"]:.4f}, R2CV={best_r2cv[0]:.4f} | '
                  f'L1={params[1]}, L2={params[2]}, Act={params[3]}, '
                  f'L2_reg={params[4]:.6f}, L1_reg={params[5]:.6f}, Drop={params[6]:.3f}')
        return target

    print(f'  Running DE for regularized MLP (popsize=2, maxiter=5, ~84 evaluations)...')

    res = differential_evolution(
        objective,
        bounds,
        popsize=2,
        maxiter=5,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=1,
        polish=False,
        disp=False
    )

    print(f'  DE-Reg complete: {res.nfev} evaluations, best R2CV={best_r2cv[0]:.4f}')
    print(f'  Computing final model with best params...')
    best_x = res.x
    best_params = decode_params(best_x)
    target, output = SumSqr_Reg(best_params, Pred, Resp, cvss)

    Mdl = output['Mdl']
    A1 = {
        'NumLayers': best_params[0],
        'Layer_1': best_params[1],
        'Layer_2': best_params[2],
        'Activation': best_params[3],
        'Alpha_L2': best_params[4],
        'Alpha_L1': best_params[5],
        'Dropout': best_params[6],
        'R2': output['R2'],
        'R2CV': output['R2CV'],
        'DE_nfev': res.nfev,
        'DE_convergence': convergence_history
    }

    return Mdl, A1