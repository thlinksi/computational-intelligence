import requests
import matplotlib.pyplot as plt

BASE_URL = "https://ooi-lab1.up.railway.app"
SIGNAL_PATH = "/api/sensor?step={step}"
AVERAGE_COUNT = 4
OUTLIER_THRESHOLD = 3.0
session = requests.Session()

def get_json(path):
    response = session.get(BASE_URL + path, timeout=10)
    response.raise_for_status()
    return response.json()
def extract_list(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return value
    return None
def get_signal(step):
    data = get_json(SIGNAL_PATH.format(step=step))
    result = extract_list(data)
    if result is None:
        raise ValueError("У відповіді немає масиву: " + str(data)[:200])
    return result
def cross_correlation(signal, template):
    n = len(signal)
    m = len(template)
    result = []
    for k in range(n - m + 1):
        total = 0.0
        for j in range(m):
            total += signal[k + j] * template[j]
        result.append(total)
    return result
def find_peak(correlation):
    best_k = 0
    best_value = correlation[0]
    for k in range(1, len(correlation)):
        if correlation[k] > best_value:
            best_value = correlation[k]
            best_k = k
    return best_k
def shift_to_depth(k, sample_rate, sound_speed):
    delay = k / sample_rate
    return delay * sound_speed / 2
def average_signals(signals):
    count = len(signals)
    length = len(signals[0])
    averaged = []
    for i in range(length):
        total = 0.0
        for s in signals:
            total += s[i]
        averaged.append(total / count)
    return averaged
def measure_depth(step, template, cfg, repeats):
    signals = [get_signal(step) for _ in range(repeats)]
    signal = average_signals(signals) if repeats > 1 else signals[0]
    correlation = cross_correlation(signal, template)
    k = find_peak(correlation)
    depth = shift_to_depth(k, cfg["sample_rate"], cfg["sound_speed"])
    return depth, k
def build_profile(template, cfg, repeats, verbose=False):
    profile = []
    total = cfg["total_steps"]
    for step in range(total):
        depth, k = measure_depth(step, template, cfg, repeats)
        profile.append(depth)

        if verbose:
            print(f"Крок {step}: Зсув = {k}, Глибина = {depth:.2f} м")
        else:
            if step % 10 == 0 or step == total - 1:
                print(f"\r  [~] Обробка: пройдено {step + 1}/{total} точок...", end="", flush=True)
    if not verbose:
        print("\r  [+] Сканування успішно завершено!                  ")
    return profile
def roughness(profile):
    total = 0.0
    for i in range(1, len(profile)):
        total += abs(profile[i] - profile[i - 1])
    return total / (len(profile) - 1)
def count_outliers(profile):
    count = 0
    for i in range(1, len(profile) - 1):
        left = abs(profile[i] - profile[i - 1])
        right = abs(profile[i] - profile[i + 1])
        if left > OUTLIER_THRESHOLD and right > OUTLIER_THRESHOLD:
            count += 1
    return count

cfg = get_json("/api/config")
template = get_json("/api/pulse")
print("Конфіг:", cfg)
print("Довжина імпульсу:", len(template), "відліків")
print("\nОдин запит на крок:")
profile_single = build_profile(template, cfg, 1, verbose=False)
print(f"\nУсереднення {AVERAGE_COUNT} запитів на крок:")
profile_avg = build_profile(template, cfg, AVERAGE_COUNT, verbose=True)
max_single = max(profile_single)
max_avg = max(profile_avg)
n_samples = len(get_signal(0))
theory_max = shift_to_depth(n_samples - len(template),
                            cfg["sample_rate"], cfg["sound_speed"])
print(f"Макс. глибина (1 запит):  {max_single:.2f} м")
print(f"Макс. глибина (усеред.):  {max_avg:.2f} м")
print(f"Теоретична межа системи:  {theory_max:.2f} м")
print(f"Шорсткість:  1 запит = {roughness(profile_single):.3f},  усереднення = {roughness(profile_avg):.3f}")
print(f"Викидів:     1 запит = {count_outliers(profile_single)},  усереднення = {count_outliers(profile_avg)}")

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)
axes[0].plot(profile_single, color="tab:red")
axes[0].set_title("Профіль дна: 1 запит на крок")
axes[1].plot(profile_avg, color="tab:blue")
axes[1].set_title(f"Профіль дна: усереднення {AVERAGE_COUNT} запитів")
for ax in axes:
    ax.invert_yaxis()
    ax.set_ylabel("Глибина, м")
    ax.grid(True)
axes[1].set_xlabel("Крок маршруту")
plt.tight_layout()
plt.show()