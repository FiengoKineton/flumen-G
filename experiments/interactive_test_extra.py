import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from flumen import CausalFlowModel
from flumen.utils import pack_model_inputs
from generate_data import make_trajectory_sampler

from argparse import ArgumentParser

import yaml
from pathlib import Path
import sys
from pprint import pprint
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.metrics import structural_similarity as ssim
from scipy.fft import fft
from scipy.signal import correlate
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf



r"""
VDP
- (old) python.exe .\experiments\interactive_test_extra.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-x3zk3ip4:v0 --note old
- (lpv) python.exe .\experiments\interactive_test_extra.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-lbz1tnpu:v3 --note lpv
- (stc) python.exe .\experiments\interactive_test_extra.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-lwqp2l3z:v3 --note static


FHN
- (old) python.exe .\experiments\interactive_test_extra.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-w52hqxkd:v6 --note old
- (new) python.exe .\experiments\interactive_test_extra.py C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\paths\fhn-upd_steps\1 --note new


NAD_big
- (old) python.exe .\experiments\interactive_test_extra.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-jwlwuqmw:v10 --note old_big
- (new) python.exe .\experiments\interactive_test_extra.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-3dxiz9gf:v2 --note new_big


NAD_small
- (old) python.exe .\experiments\interactive_test_extra.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-mg4z6swx:v10 --note old_small
- (new) python.exe .\experiments\interactive_test_extra.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-zshs5333:v0 --note new_small
"""

ERROR = False
PHASE = True
FFT = False
CORRELATION = False


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file" "(or, if run with --wandb, path to a Weights & Biases artifact)")
    ap.add_argument('--print_info', action='store_true', help="Print training metadata and quit")
    ap.add_argument('--continuous_state', action='store_true')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--time_horizon', type=float, default=None)
    ap.add_argument('--more', action='store_true')
    ap.add_argument('--note', type=str, default=None)

    return ap.parse_args()




def compute_ssim_phase(y_true, y_pred):
    """Compute SSIM between phase portraits (2D histograms)"""
    assert y_true.shape[1] == 2, "Only 2D phase space supported"

    def to_hist_img(y, bins=64):
        hist, _, _ = np.histogram2d(y[:, 0], y[:, 1], bins=bins, density=True)
        hist = hist / hist.max()
        return hist

    true_img = to_hist_img(y_true)
    pred_img = to_hist_img(y_pred)
    ssim_val = ssim(true_img, pred_img, data_range=1.0)
    return ssim_val


def compute_fft_mismatch(y_true, y_pred):
    """Compare FFT magnitude spectra"""
    assert y_true.shape[1] == y_pred.shape[1]

    errors = []
    for i in range(y_true.shape[1]):
        true_fft = np.abs(fft(y_true[:, i]))
        pred_fft = np.abs(fft(y_pred[:, i]))
        true_fft /= true_fft.max()
        pred_fft /= pred_fft.max()

        mismatch = np.mean((true_fft - pred_fft) ** 2)
        errors.append(mismatch)

    return errors


def compute_error_growth(y_true, y_pred):
    """Qualitative: how the error grows over time"""
    err_t = np.mean((y_true - y_pred) ** 2, axis=1)
    return err_t

def compute_lyap_like(y_true, y_pred, eps=1e-8):
    err_t = np.mean((y_true - y_pred) ** 2, axis=1)
    return np.log(err_t + eps)  # log error to mimic Lyapunov rate


def evaluate_all(y_true, y_pred, t, NAME, N, NOTE=None):
    print("=== Extra Metrics ===")


    if t is not None and ERROR:
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        # === Error growth over time ===
        # Il primo subplot mostra l'errore quadratico medio (MSE) nel tempo.
        # ➤ Se la curva è bassa e piatta → il modello è stabile e preciso nel lungo termine.
        # ➤ Se ci sono picchi o crescite improvvise → il modello diverge o sbaglia in certe regioni.

        # Il secondo subplot mostra il logaritmo dell'errore, utile per osservare la tendenza esponenziale.
        # ➤ Una crescita lineare nel log(MSE) suggerisce una divergenza sistemica (stile esponente di Lyapunov).
        # ➤ Se la curva si stabilizza o decresce → il modello è robusto.


        # First plot: MSE
        err_curve = compute_error_growth(y_true, y_pred)
        axes[0].plot(t, err_curve, label="Error growth", color="red")
        axes[0].set_ylabel("MSE")
        axes[0].legend()
        axes[0].grid()
        axes[0].set_title("Error vs Time")

        # Second plot: Log-MSE
        err_curve_log = compute_lyap_like(y_true, y_pred) 
        axes[1].plot(t, err_curve_log, label="Error log-growth", color="red")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("log(MSE)")
        axes[1].legend()
        axes[1].grid()
        axes[1].set_title("Log Error vs Time")


        if NOTE:    plt.suptitle(f"{NAME} error growth -- {NOTE}")
        else:       plt.suptitle(f"{NAME} error growth")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.show()

    if CORRELATION: 
        if N==2:
            fig, axes = plt.subplots(y_true.shape[1], 1, figsize=(8, 3 * y_true.shape[1]))
            lags=100
            for i in range(y_true.shape[1]):
                acf_true = acf(y_true[:, i], nlags=lags)
                acf_pred = acf(y_pred[:, i], nlags=lags)
                ax = axes[i] if y_true.shape[1] > 1 else axes
                ax.plot(acf_true, label='True ACF', color='blue')
                ax.plot(acf_pred, label='Pred ACF', color='orange', linestyle='--')
                ax.set_title(f'$x_{{{i+1}}}$')
                ax.legend()
                ax.grid(True)
            plt.tight_layout()
            if NOTE:    plt.suptitle(f"{NAME} Autocorrelation -- {NOTE}")
            else:       plt.suptitle(f"{NAME} Autocorrelation")
            plt.tight_layout()
        else: 
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
            lags=100
            i = N-1
            acf_true = acf(y_true[:, i], nlags=lags)
            acf_pred = acf(y_pred[:, i], nlags=lags)
            ax.plot(acf_true, label='True ACF', color='blue')
            ax.plot(acf_pred, label='Pred ACF', color='orange', linestyle='--')
            ax.set_title(f'$x_{{{i+1}}}$')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            if NOTE:    plt.suptitle(f"{NAME} Autocorrelation -- {NOTE}")
            else:       plt.suptitle(f"{NAME} Autocorrelation")
            plt.tight_layout()



    if PHASE:
        # === Phase Portrait Comparison ===
        # A sinistra: distribuzione del sistema reale nel piano delle fasi (x1 vs x2).
        # A destra: distribuzione predetta dal modello.
        # ➤ Se le due immagini hanno forma e densità simili → il modello ha appreso la struttura dinamica.
        # ➤ Differenze marcate indicano che il modello ha generalizzato male o ha perso la topologia originale.

        # L'SSIM misura la similarità strutturale: 1 = identici, 0 = completamente diversi.
        # ➤ Un SSIM > 0.6 è generalmente buono in contesti dinamici.
        
        if N == 2: idx = [0, 1]
        elif N==100: idx = [0, -1]
        else: idx = [N-2, N-1]
        ssim_val = compute_ssim_phase(y_true[:,idx], y_pred[:,idx])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist2d(y_true[:, idx[0]], y_true[:, idx[1]], bins=64, density=True, cmap='viridis')
        axes[0].set_title("True Phase Portrait")
        axes[0].set_xlabel("$x_1$")
        axes[0].set_ylabel("$x_2$")

        axes[1].hist2d(y_pred[:, idx[0]], y_pred[:, idx[1]], bins=64, density=True, cmap='plasma')
        axes[1].set_title("Predicted Phase Portrait")
        axes[1].set_xlabel(f"$x_{{{idx[0]}}}$")
        axes[1].set_ylabel(f"$x_{{{idx[1]}}}$")


        if NOTE:    plt.suptitle(f"{NAME} SSIM (phase portrait): {ssim_val:.4f} -- {NOTE}")
        else:       plt.suptitle(f"{NAME} SSIM (phase portrait): {ssim_val:.4f}")
        plt.tight_layout()
        #plt.show()



    if FFT:
        # === Frequency Spectrum Comparison ===
        # Per ogni variabile (x1, x2, ...), viene confrontato lo spettro in frequenza:
        # - Blu = spettro reale
        # - Arancione = spettro predetto

        # ➤ Se le due curve si sovrappongono → il modello cattura la dinamica armonica (oscillazioni, periodicità).
        # ➤ Divergenze (soprattutto nelle basse frequenze) indicano che la dinamica fondamentale non è stata appresa.

        # L’errore FFT (mismatch) deve essere il più basso possibile.
        # ➤ < 0.001 → ottima compatibilità spettrale.

        # === FFT COMPARISON ===
        fft_err = compute_fft_mismatch(y_true, y_pred)
        error = np.zeros(N)
        for i, err in enumerate(fft_err):
            #print(f"FFT Mismatch (x{i+1}): {err:.6f}")
            error[i] = err

        if N==2:
            fig, axes = plt.subplots(y_true.shape[1], 1, figsize=(8, 3 * y_true.shape[1]))
            for i in range(y_true.shape[1]):
                true_fft = np.abs(fft(y_true[:, i]))
                pred_fft = np.abs(fft(y_pred[:, i]))
                true_fft /= true_fft.max()
                pred_fft /= pred_fft.max()

                freqs = np.fft.fftfreq(len(true_fft), d=(t[1] - t[0]))  # assume uniform t
                mask = freqs >= 0

                ax = axes[i] if y_true.shape[1] > 1 else axes
                ax.plot(freqs[mask], true_fft[mask], label="True", color="blue")
                ax.plot(freqs[mask], pred_fft[mask], label="Predicted", color="orange", linestyle="--")
                ax.set_title(f"FFT Comparison for $x_{i+1}$: {error[i]:.6f}")
                ax.set_xlabel("Frequency")
                ax.set_ylabel("Normalized Magnitude")
                ax.set_yscale("log")
                ax.legend()
                ax.grid(True)
        else: 
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
            i = N-1
            true_fft = np.abs(fft(y_true[:, i]))
            pred_fft = np.abs(fft(y_pred[:, i]))
            true_fft /= true_fft.max()
            pred_fft /= pred_fft.max()

            freqs = np.fft.fftfreq(len(true_fft), d=(t[1] - t[0]))  # assume uniform t
            mask = freqs >= 0

            ax.plot(freqs[mask], true_fft[mask], label="True", color="blue")
            ax.plot(freqs[mask], pred_fft[mask], label="Predicted", color="orange", linestyle="--")
            ax.set_title(f"FFT Comparison for $x_{{{i+1}}}$: {error[i]:.6f}")
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Normalized Magnitude")
            ax.set_yscale("log")
            ax.legend()
            ax.grid(True)

        if NOTE:    plt.suptitle(f"{NAME} FFT -- {NOTE}")
        else:       plt.suptitle(f"{NAME} FFT")
        plt.tight_layout()
    
    plt.show()


# Example usage (replace these arrays with yours):
if __name__ == "__main__":
    from numpy import load

    #plt.ion()
    args = parse_args()

    num_times = 2 if args.time_horizon is None else 1          # default 2 | multiplied by the time_span
    more = args.more


    if args.wandb:
        import wandb
        api = wandb.Api()
        model_artifact = api.artifact(args.path)        
        model_path = Path(model_artifact.download())

        model_run = model_artifact.logged_by()
        print(model_run.summary)
    else:
        model_path = Path(args.path)

    with open(model_path / "state_dict.pth", 'rb') as f:
        state_dict = torch.load(f, weights_only=True, map_location='cpu')
    with open(model_path / "metadata.yaml", 'r') as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)

    pprint(metadata)

    model = CausalFlowModel(**metadata["args"])
    model.load_state_dict(state_dict)
    model.eval()

    sampler = make_trajectory_sampler(metadata["data_settings"])
    sampler.reset_rngs()
    delta = sampler._delta

    if args.continuous_state:
        xx = np.linspace(0., 1., model.state_dim)
        n_plots = 2
    else:
        n_plots = model.output_dim
    
    num = n_plots if n_plots==2 else 1
    time_horizon = args.time_horizon if args.time_horizon else num_times * metadata["data_args"]["time_horizon"]


    # --------------------------------------------------------
    while True:
        time_integrate = time()
        x0, t, y, u = sampler.get_example(time_horizon=time_horizon, n_samples=int(1 + 100 * time_horizon))
        time_integrate = time() - time_integrate

        time_predict = time()
        x0_feed, t_feed, u_feed, deltas_feed = pack_model_inputs(x0, t, u, delta)

        with torch.no_grad():
            y_pred, _, _ = model(x0_feed, u_feed, deltas_feed)
            # model 2 

        y_pred = np.flip(y_pred.numpy(), 0)
        time_predict = time() - time_predict

        print(f"Timings: {time_integrate}, {time_predict}")

        y = y[:, tuple(bool(v) for v in sampler._dyn.mask)]
        sq_error = model.output_dim * np.mean(np.square(y - y_pred))
        print("MSE (mean square error):", sq_error, "\n")


        NAME = metadata['data_settings']['dynamics']['name']
        N = model.output_dim
        NOTE = args.note

        # ax1[0].set_title(f'{NAME} dynamics ($x \\in \\mathcal{{R}}^{{{N}}}$) -- {NOTE}')


        # --------------------------------------------------------
        # Example:
        print(f'state dimension: {N}')
        evaluate_all(y, y_pred, t, NAME, N, NOTE)