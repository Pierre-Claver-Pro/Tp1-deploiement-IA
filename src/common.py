import time, json, os, threading
import numpy as np
import torch

DRIVE = '/content/drive/MyDrive/tp_deploy'
BS, N_RUNS, WARMUP, REPS = 10, 5, 3, 5   # batch, runs indépendants, warmup, passes par run


def load_data():
    d = torch.load(f'{DRIVE}/inputs.pt')
    return d['X'], d['y'].numpy()


def load_model():
    import torchvision
    m = torchvision.models.resnet18()
    m.fc = torch.nn.Linear(512, 10)
    m.load_state_dict(torch.load(f'{DRIVE}/resnet18_cifar10.pt', map_location='cpu'))
    return m.eval()


def hw_info():
    import platform, subprocess
    info = {'cpu': platform.processor() or platform.machine(), 'cores': os.cpu_count(), 'torch': torch.__version__}
    try:
        info['cpu'] = [l for l in open('/proc/cpuinfo') if 'model name' in l][0].split(':')[1].strip()
    except Exception:
        pass
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
    return info


def _stats(v):
    v = np.asarray(v, dtype=float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()))


class EnergyMeter:
    """GPU : puissance échantillonnée via NVML. CPU : compteur RAPL si accessible (souvent indisponible sur Colab)."""
    def __init__(self, device):
        self.device, self.ok, self.joules, self.watts = device, False, None, None
        if device == 'gpu':
            try:
                import pynvml
                pynvml.nvmlInit(); self.nv = pynvml
                self.h = pynvml.nvmlDeviceGetHandleByIndex(0); self.ok = True
            except Exception as e:
                print('NVML indisponible :', e)
        else:
            self.rapl = '/sys/class/powercap/intel-rapl:0/energy_uj'
            try:
                open(self.rapl).read(); self.ok = True
            except Exception as e:
                print('RAPL indisponible :', e)

    def __enter__(self):
        self.t0 = time.perf_counter()
        if self.ok and self.device == 'gpu':
            self.samples, self._stop = [], False
            def loop():
                while not self._stop:
                    self.samples.append(self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000)
                    time.sleep(0.02)
            self.th = threading.Thread(target=loop); self.th.start()
        elif self.ok:
            self.e0 = int(open(self.rapl).read())
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
        if not self.ok:
            return
        if self.device == 'gpu':
            self._stop = True; self.th.join()
            self.watts = float(np.mean(self.samples)); self.joules = self.watts * self.dt
        else:
            j = (int(open(self.rapl).read()) - self.e0) / 1e6
            if j > 0:
                self.joules, self.watts = j, j / self.dt


def run_benchmark(name, predict, X, y, device, energy_seconds=10, extra=None):
    """predict(batch_tensor_cpu) -> logits numpy (B,10). Le transfert H2D/D2H est inclus dans la mesure."""
    batches = [X[i:i + BS] for i in range(0, len(X), BS)]

    def one_pass():
        out, lat = [], []
        for b in batches:
            t = time.perf_counter(); out.append(predict(b)); lat.append(time.perf_counter() - t)
        return np.concatenate(out), lat

    for _ in range(WARMUP):
        one_pass()

    runs = []
    for _ in range(N_RUNS):
        lats = []
        t = time.perf_counter()
        for _ in range(REPS):
            logits, lat = one_pass(); lats += lat
        total = time.perf_counter() - t
        preds = logits.argmax(1)
        runs.append(dict(total_s=total / REPS, throughput=REPS * len(X) / total,
                         batch_ms=1000 * float(np.mean(lats)), top1=float((preds == y).mean()),
                         preds=preds.tolist()))

    n_img = 0
    with EnergyMeter(device) as em:
        while time.perf_counter() - em.t0 < energy_seconds:
            one_pass(); n_img += len(X)
    energy = None
    if em.joules is not None:
        energy = dict(watts=em.watts, joules_per_image=em.joules / n_img)

    summary = {k: _stats([r[k] for r in runs]) for k in ['total_s', 'throughput', 'batch_ms', 'top1']}
    res = dict(name=name, device=device, bs=BS, n_runs=N_RUNS, reps=REPS, summary=summary, energy=energy,
               preds=runs[0]['preds'], preds_stable=all(r['preds'] == runs[0]['preds'] for r in runs),
               hw=hw_info(), extra=extra or {})
    os.makedirs(f'{DRIVE}/results', exist_ok=True)
    json.dump(res, open(f'{DRIVE}/results/{name}.json', 'w'))
    for k, s in summary.items():
        print(f"{k:12s} moy={s['mean']:.4f} std={s['std']:.4f} min={s['min']:.4f} max={s['max']:.4f}")
    print('énergie :', energy)
    return res
