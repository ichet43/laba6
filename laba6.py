# ============================================================
# ЛАБОРАТОРНАЯ РАБОТА — ВАРИАНТ 49: КАМЕРЫ / ТРАФИК
# ============================================================

import numpy as np

np.set_printoptions(suppress=True, precision=4)

CSV_PATH = "data.csv"
OUT_BIN = "traffic_final.npy"

dt = np.dtype([
    ('ts', 'i4'),
    ('cam_id', 'i2'),
    ('veh', 'u2'),
    ('wait', 'f4'),
    ('viol_r', 'f4'),
    ('cong', 'f4'),
])


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def add_fields(data, fields):
    """Добавление новых полей в структурированный массив."""

    new_dtype = np.dtype(
        data.dtype.descr +
        [(name, dtype) for name, dtype, _ in fields]
    )

    result = np.empty(data.shape, dtype=new_dtype)

    for name in data.dtype.names:
        result[name] = data[name]

    for name, dtype, values in fields:
        result[name] = values

    return result


def sanitize(arr):
    """Замена NaN и Inf медианой конечных значений."""

    arr = np.asarray(arr, dtype=np.float64).copy()

    finite = np.isfinite(arr)

    median = (
        float(np.median(arr[finite]))
        if finite.any()
        else 0.0
    )

    arr[~finite] = median

    return arr


def moving_average(x, k):
    """Скользящее среднее через кумулятивную сумму O(N)."""

    x = np.asarray(x, dtype=np.float64)

    if len(x) == 0:
        return np.array([], dtype=np.float64)

    if len(x) < k:
        return np.full(len(x), x.mean())

    csum = np.cumsum(x, dtype=np.float64)

    csum_pad = np.concatenate(([0.0], csum))

    ma = (
        csum_pad[k:] - csum_pad[:-k]
    ) / k

    return np.pad(
        ma,
        (k - 1, 0),
        mode='edge'
    )


# ============================================================
# ЭТАП 1. ЗАГРУЗКА И ПОДГОТОВКА ТИПОВ
# ============================================================

print("=" * 60)
print("ЭТАП 1. Загрузка и подготовка типов")
print("=" * 60)


def load_csv_lines(path, dt, encoding="utf-8-sig"):

    names = list(dt.names)
    n_fields = len(names)

    fields = {name: [] for name in names}

    dropped = 0

    with open(path, encoding=encoding, newline="") as f:

        header = f.readline().rstrip("\r\n").split(",")

        if header != names:
            print(
                f"⚠️ Заголовок: {header}\n"
                f"Ожидался: {names}\n"
                "Чтение выполняется по позициям."
            )

        for line in f:

            parts = line.rstrip("\r\n").split(",")

            if len(parts) != n_fields:
                dropped += 1
                continue

            try:
                values = [float(p) for p in parts]
            except ValueError:
                dropped += 1
                continue

            for i, name in enumerate(names):
                fields[name].append(values[i])

    # Исходные числовые столбцы float64.
    # Проверка аномалий выполняется ДО преобразования типов.

    cols = {
        name: np.array(fields[name], dtype=np.float64)
        for name in names
    }

    n_rows = len(cols['ts'])

    if n_rows == 0:
        raise ValueError("CSV не содержит корректных записей.")

    print("\nПроверка NaN и Inf до очистки:")

    for name in names:

        arr = cols[name]

        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())

        bad_count = nan_count + inf_count
        share = bad_count / n_rows

        print(
            f"  {name:>7}: "
            f"NaN={nan_count}, "
            f"Inf={inf_count}, "
            f"доля={share * 100:.2f}%"
        )

        if share > 0.03:
            print(
                f"  ⚠️ ВНИМАНИЕ: в поле {name} "
                "более 3% NaN/Inf!"
            )

    # Проверяем отрицательные veh до преобразования в uint16.

    veh_negative = cols['veh'] < 0

    print(
        f"\nОтрицательных veh до преобразования: "
        f"{veh_negative.sum()}"
    )

    # Очистка целочисленных полей перед приведением типов.

    for name in ['ts', 'cam_id']:

        arr = cols[name]
        finite = np.isfinite(arr)

        median = (
            float(np.median(arr[finite]))
            if finite.any()
            else 0.0
        )

        arr = np.where(finite, arr, median)
        arr = np.rint(arr)

        info = np.iinfo(dt[name])

        cols[name] = np.clip(
            arr,
            info.min,
            info.max
        )

    # veh: отрицательные, NaN, Inf -> 0.
    # Значения выше uint16 ограничиваем максимумом.

    veh = cols['veh']

    veh = np.where(
        np.isfinite(veh) & (veh >= 0),
        veh,
        0.0
    )

    veh = np.clip(veh, 0, 65535)

    cols['veh'] = np.rint(veh)

    # Формирование структурированного массива.

    data = np.empty(n_rows, dtype=dt)

    for name in names:
        data[name] = cols[name].astype(dt[name])

    print(f"\nЗагружено строк: {n_rows}")
    print(f"Отброшено некорректных строк: {dropped}")

    return data, int(veh_negative.sum())


data, veh_negative_count = load_csv_lines(
    CSV_PATH,
    dt
)

n_rows = len(data)

print(f"\nКоличество строк: {n_rows}")

print(
    f"Объём памяти: {data.nbytes} байт "
    f"= {data.nbytes / 1024 / 1024:.3f} МБ"
)

print(f"Размер записи: {data.dtype.itemsize} байт")
print(f"dtype: {data.dtype}")

np.save("traffic_v49_raw.npy", data)

print("\nСохранено: traffic_v49_raw.npy\n")


# ============================================================
# ЭТАП 2. ВЕКТОРИЗОВАННАЯ ФИЛЬТРАЦИЯ И ОЧИСТКА
# ============================================================

print("=" * 60)
print("ЭТАП 2. Векторизованная фильтрация и очистка")
print("=" * 60)

data = data.copy()

# veh уже имеет тип uint16.
# Отрицательные значения обнаружены и исправлены при загрузке.

mask_wait_neg = data['wait'] < 0

mask_viol_bad = (
    (data['viol_r'] < 0) |
    (data['viol_r'] > 1)
)

mask_wait_nonfinite = ~np.isfinite(data['wait'])
mask_viol_nonfinite = ~np.isfinite(data['viol_r'])
mask_cong_nonfinite = ~np.isfinite(data['cong'])

mask_anom = (
    mask_wait_neg |
    mask_viol_bad |
    mask_wait_nonfinite |
    mask_viol_nonfinite |
    mask_cong_nonfinite
)

print(
    f"Строк с аномалиями после загрузки: "
    f"{mask_anom.sum()} "
    f"({mask_anom.mean() * 100:.2f}%)"
)

print(f"  veh<0 при чтении CSV: {veh_negative_count}")
print(f"  wait<0: {mask_wait_neg.sum()}")
print(f"  viol_r вне [0,1]: {mask_viol_bad.sum()}")
print(f"  NaN/Inf wait: {mask_wait_nonfinite.sum()}")
print(f"  NaN/Inf viol_r: {mask_viol_nonfinite.sum()}")
print(f"  NaN/Inf cong: {mask_cong_nonfinite.sum()}")

# Очистка wait.

data['wait'] = np.where(
    np.isfinite(data['wait']),
    data['wait'],
    0.0
)

data['wait'] = np.where(
    data['wait'] < 0,
    0.0,
    data['wait']
).astype('f4')

# Очистка viol_r.

data['viol_r'] = np.where(
    np.isfinite(data['viol_r']),
    data['viol_r'],
    0.0
)

data['viol_r'] = np.clip(
    data['viol_r'],
    0.0,
    1.0
).astype('f4')

# Очистка cong.
# Пока заменяем только NaN/Inf.
# Нормализация выполняется на этапе 3.

finite_cong = data['cong'][
    np.isfinite(data['cong'])
]

cong_med = (
    float(np.median(finite_cong))
    if finite_cong.size
    else 0.0
)

data['cong'] = np.where(
    np.isfinite(data['cong']),
    data['cong'],
    cong_med
).astype('f4')

print("\nПосле очистки:")

for name in ['wait', 'viol_r', 'cong']:

    print(
        f"  {name}: "
        f"min={data[name].min():.4f}, "
        f"max={data[name].max():.4f}"
    )

print()


# ============================================================
# ЭТАП 3. ГРУППИРОВКА И НОРМАЛИЗАЦИЯ
# ============================================================

print("=" * 60)
print("ЭТАП 3. Группировка и нормализация cong")
print("=" * 60)

cams, inv, counts = np.unique(
    data['cam_id'],
    return_inverse=True,
    return_counts=True
)

n_groups = len(cams)

print(f"Количество камер: {n_groups}")

# ------------------------------------------------------------
# 3.1. Группировка: mean(wait), max(veh)
# ------------------------------------------------------------

mean_wait = np.empty(n_groups, dtype=np.float64)
max_veh = np.empty(n_groups, dtype=np.float64)

for gi in range(n_groups):

    mask = inv == gi

    mean_wait[gi] = data['wait'][mask].mean()
    max_veh[gi] = data['veh'][mask].max()

print("\ncam_id | count | mean_wait | max_veh")

for c, n, mw, mv in zip(
    cams,
    counts,
    mean_wait,
    max_veh
):

    print(
        f"{int(c):>6} | "
        f"{n:>5} | "
        f"{mw:>9.3f} | "
        f"{int(mv):>7}"
    )

# ------------------------------------------------------------
# 3.2. Min-Max нормализация cong по камерам
# ------------------------------------------------------------

cong_norm = np.empty(
    n_rows,
    dtype=np.float64
)

for gi in range(n_groups):

    mask = inv == gi

    values = data['cong'][mask].astype(np.float64)

    vmin = values.min()
    vmax = values.max()

    if vmax > vmin:
        cong_norm[mask] = (
            (values - vmin) /
            (vmax - vmin)
        )
    else:
        cong_norm[mask] = 0.0

# Ограничение нормализованного показателя.

cong_norm = np.clip(
    cong_norm,
    0.0,
    1.0
)

print("\nMin-Max нормализация cong выполнена.")

# ------------------------------------------------------------
# 3.3. Z-score по камерам
# ------------------------------------------------------------

cong_z = np.empty(
    n_rows,
    dtype=np.float64
)

mean_cong = np.empty(n_groups)
std_cong = np.empty(n_groups)

for gi in range(n_groups):

    mask = inv == gi

    values = cong_norm[mask]

    mean_cong[gi] = values.mean()
    std_cong[gi] = values.std()

    if std_cong[gi] > 1e-12:

        cong_z[mask] = (
            (values - mean_cong[gi]) /
            std_cong[gi]
        )

    else:
        cong_z[mask] = 0.0

print("\nСтатистика нормализованного cong:")

print("cam_id | mean | std")

for c, mean, std in zip(
    cams,
    mean_cong,
    std_cong
):

    print(
        f"{int(c):>6} | "
        f"{mean:>6.4f} | "
        f"{std:>6.4f}"
    )

print("\nПроверка Z-score:")

for gi, c in enumerate(cams):

    mask = inv == gi

    print(
        f"cam={int(c):>3}: "
        f"mean={cong_z[mask].mean():+.4f}, "
        f"std={cong_z[mask].std():.4f}"
    )

# Добавление нормализованных полей.

data = add_fields(data, [
    ('cong_norm', 'f8', cong_norm),
    ('cong_z', 'f8', cong_z)
])

np.save("traffic_normalized.npy", data)

print("\nСохранено: traffic_normalized.npy\n")


# ============================================================
# ЭТАП 4. СКОЛЬЗЯЩЕЕ ОКНО k=45 И np.diff(cong)
# ============================================================

print("=" * 60)
print("ЭТАП 4. Скользящее окно k=45 и разница cong")
print("=" * 60)

k = 45

wait_values = data['wait'].astype(np.float64)

moving_avg = moving_average(
    wait_values,
    k
)

print(
    f"Moving average wait: "
    f"k={k}, длина={len(moving_avg)}"
)

# Разница индекса загруженности.

cong_values = data['cong'].astype(np.float64)

diff_cong = np.diff(
    cong_values,
    prepend=cong_values[0]
)

print(
    f"np.diff(cong): "
    f"min={diff_cong.min():.4f}, "
    f"max={diff_cong.max():.4f}"
)

# Индекс резкого роста загруженности.

threshold = diff_cong.std()

cong_spike = (
    diff_cong > threshold
).astype(np.int8)

print(f"Порог роста загруженности: {threshold:.4f}")
print(f"Количество скачков: {cong_spike.sum()}")

data = add_fields(data, [
    ('moving_avg', 'f8', moving_avg),
    ('diff_cong', 'f8', diff_cong),
    ('cong_spike', 'i1', cong_spike)
])

print()


# ============================================================
# ЭТАП 5. FEATURE ENGINEERING
# ============================================================

print("=" * 60)
print("ЭТАП 5. Производные признаки")
print("=" * 60)

wait_f = data['wait'].astype(np.float64)
veh_f = data['veh'].astype(np.float64)
viol_f = data['viol_r'].astype(np.float64)

# Среднее ожидание на автомобиль.
# При veh=0 результат равен 0.

wait_per_veh = np.where(
    veh_f > 0,
    wait_f / np.maximum(veh_f, 1.0),
    0.0
)

# Оценочное количество нарушений.

viol_veh = viol_f * veh_f

wait_per_veh = sanitize(wait_per_veh)
viol_veh = sanitize(viol_veh)

print(
    f"wait_per_veh: "
    f"mean={wait_per_veh.mean():.3f}, "
    f"std={wait_per_veh.std():.3f}"
)

print(
    f"viol_veh: "
    f"mean={viol_veh.mean():.3f}, "
    f"std={viol_veh.std():.3f}"
)

data = add_fields(data, [
    ('wait_per_veh', 'f8', wait_per_veh),
    ('viol_veh', 'f8', viol_veh)
])

print()


# ============================================================
# ЭТАП 6. УСЛОВНАЯ АГРЕГАЦИЯ
# ============================================================

print("=" * 60)
print("ЭТАП 6. Условная агрегация по cam_id")
print("=" * 60)

# Условие: загруженность выше общей медианы.

cond_thr = float(np.median(data['cong']))

cond = data['cong'] > cond_thr

print(
    f"Условие cong > {cond_thr:.4f}: "
    f"{cond.sum()} строк "
    f"({cond.mean() * 100:.2f}%)"
)

rows = []
p90_values = []

for gi, c in enumerate(cams):

    # Составная булева маска.

    mask = (inv == gi) & cond

    if not mask.any():

        rows.append([
            float(c),
            np.nan,
            np.nan
        ])

        p90_values.append(np.nan)

        continue

    values = data['wait'][mask]

    mean_value = float(values.mean())
    median_value = float(np.median(values))
    p90_value = float(np.percentile(values, 90))

    rows.append([
        float(c),
        mean_value,
        median_value
    ])

    p90_values.append(p90_value)

# Форма строго (N_groups, 3).

cond_stats = np.array(
    rows,
    dtype=np.float64
)

p90_values = np.array(
    p90_values,
    dtype=np.float64
)

print("\ncam_id | mean_wait | median_wait | P90")

for row, p90 in zip(
    cond_stats,
    p90_values
):

    print(
        f"{int(row[0]):>6} | "
        f"{row[1]:>9.3f} | "
        f"{row[2]:>11.3f} | "
        f"{p90:>8.3f}"
    )

print(
    f"\nФорма cond_stats: {cond_stats.shape}"
)

np.save("traffic_cond_stats.npy", cond_stats)

print()


# ============================================================
# ЭТАП 7. ЛАГОВЫЕ ПРИЗНАКИ
# ============================================================

print("=" * 60)
print("ЭТАП 7. Лаговые признаки wait")
print("=" * 60)

wait_values = data['wait'].astype(np.float64)

lag1 = np.roll(wait_values, 1)

lag1[0] = wait_values[0]

delta_lag = wait_values - lag1

print(
    f"delta_wait: "
    f"min={delta_lag.min():.3f}, "
    f"max={delta_lag.max():.3f}"
)

up = int(np.sum(delta_lag > 0))
down = int(np.sum(delta_lag < 0))
flat = int(np.sum(delta_lag == 0))

total = len(delta_lag)

print(f"Выросло: {up} ({up / total * 100:.2f}%)")
print(f"Упало: {down} ({down / total * 100:.2f}%)")
print(f"Без изменений: {flat} ({flat / total * 100:.2f}%)")

signs = np.sign(delta_lag).astype(np.int8)

counts_sign = np.bincount(
    signs + 1,
    minlength=3
)

print(
    f"np.bincount [-1, 0, +1]: "
    f"{counts_sign}"
)

data = add_fields(data, [
    ('lag1_wait', 'f8', lag1),
    ('delta_wait', 'f8', delta_lag)
])

print()


# ============================================================
# ЭТАП 8. ГРУППОВАЯ IQR-ЗАМЕНА ВЫБРОСОВ
# ============================================================

print("=" * 60)
print("ЭТАП 8. IQR-замена выбросов wait по камерам")
print("=" * 60)

wait_values = data['wait'].astype(np.float64).copy()

replaced_by_group = {}
replaced_total = 0

for gi, c in enumerate(cams):

    mask = inv == gi

    values = wait_values[mask]

    q1, q3 = np.percentile(
        values,
        [25, 75]
    )

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (values < lower) |
        (values > upper)
    )

    median = float(np.median(values))

    indices = np.where(mask)[0]

    outlier_indices = indices[outliers]

    wait_values[outlier_indices] = median

    count = len(outlier_indices)

    replaced_by_group[int(c)] = count

    replaced_total += count

print("\nКоличество замен по камерам:")

for c, count in replaced_by_group.items():

    print(
        f"  cam={c:>3}: {count}"
    )

print(
    f"\nВсего заменено: {replaced_total} "
    f"({replaced_total / n_rows * 100:.2f}%)"
)

data['wait'] = wait_values.astype('f4')

print()


# ============================================================
# ЭТАП 9. ПРОВЕРКА ЛОГИЧЕСКОЙ ЦЕЛОСТНОСТИ
# ============================================================

print("=" * 60)
print("ЭТАП 9. Проверка логической целостности")
print("=" * 60)

# Правила предметной области:
#
# R1: wait >= 0
# R2: viol_r принадлежит [0,1]
# R3: veh > 0 и wait <= 300
# R4: ts > 0

r1 = data['wait'] < 0

r2 = (
    (data['viol_r'] < 0) |
    (data['viol_r'] > 1)
)

r3 = (
    (data['veh'] == 0) |
    (data['wait'] > 300)
)

r4 = data['ts'] <= 0

viol = r1 | r2 | r3 | r4

print(
    f"Нарушений: {viol.sum()} "
    f"({viol.mean() * 100:.2f}%)"
)

print(
    f"R1 wait<0: {r1.sum()} "
    f"({r1.mean() * 100:.2f}%)"
)

print(
    f"R2 viol_r вне [0,1]: {r2.sum()} "
    f"({r2.mean() * 100:.2f}%)"
)

print(
    f"R3 veh=0 или wait>300: {r3.sum()} "
    f"({r3.mean() * 100:.2f}%)"
)

print(
    f"R4 ts<=0: {r4.sum()} "
    f"({r4.mean() * 100:.2f}%)"
)

# Медианные значения для исправления.

wait_med = float(np.median(data['wait']))
viol_med = float(np.median(data['viol_r']))

# Коррекция wait.

data['wait'] = np.where(
    r1 | r3,
    wait_med,
    data['wait']
).astype('f4')

# Коррекция viol_r.

data['viol_r'] = np.where(
    r2,
    viol_med,
    data['viol_r']
)

data['viol_r'] = np.clip(
    data['viol_r'],
    0.0,
    1.0
).astype('f4')

# Коррекция veh.

data['veh'] = np.where(
    data['veh'] == 0,
    np.uint16(1),
    data['veh']
).astype('u2')

# Коррекция ts.

ts_positive = data['ts'][
    data['ts'] > 0
]

ts_fix = (
    int(np.min(ts_positive))
    if ts_positive.size
    else 1
)

data['ts'] = np.where(
    r4,
    ts_fix,
    data['ts']
).astype('i4')

# Повторная проверка.

viol_after = (
    (data['wait'] < 0) |
    (data['viol_r'] < 0) |
    (data['viol_r'] > 1) |
    (data['veh'] == 0) |
    (data['wait'] > 300) |
    (data['ts'] <= 0)
)

print(
    f"\nНарушений после исправления: "
    f"{viol_after.sum()}"
)

print()


# ============================================================
# ПЕРЕСЧЁТ ПРОИЗВОДНЫХ ПРИЗНАКОВ ПОСЛЕ ОЧИСТКИ
# ============================================================

print("=" * 60)
print("Пересчёт производных признаков после очистки")
print("=" * 60)

# После этапов 8 и 9 изменился wait.
# После этапа 9 могли измениться veh и viol_r.
# Поэтому пересчитываем зависимые поля.

wait_f = data['wait'].astype(np.float64)
veh_f = data['veh'].astype(np.float64)
viol_f = data['viol_r'].astype(np.float64)

# Скользящее среднее.

data['moving_avg'] = moving_average(
    wait_f,
    k
)

# Производные признаки.

data['wait_per_veh'] = sanitize(
    np.where(
        veh_f > 0,
        wait_f / np.maximum(veh_f, 1.0),
        0.0
    )
)

data['viol_veh'] = sanitize(
    viol_f * veh_f
)

# Лаговые признаки.

lag1 = np.roll(wait_f, 1)

lag1[0] = wait_f[0]

data['lag1_wait'] = lag1

data['delta_wait'] = wait_f - lag1

print("Пересчитаны поля:")
print("  moving_avg")
print("  wait_per_veh")
print("  viol_veh")
print("  lag1_wait")
print("  delta_wait")

# Проверка согласованности.

check_wait_per_veh = np.where(
    veh_f > 0,
    wait_f / np.maximum(veh_f, 1.0),
    0.0
)

print(
    "\nПроверка wait_per_veh:",
    np.allclose(
        data['wait_per_veh'],
        check_wait_per_veh
    )
)

print()


# ============================================================
# ЭТАП 10. ЧАСТОТНЫЙ АНАЛИЗ И СЖАТИЕ РЕДКИХ КАТЕГОРИЙ
# ============================================================

print("=" * 60)
print("ЭТАП 10. Частотный анализ cam_id")
print("=" * 60)

cats, cat_counts = np.unique(
    data['cam_id'],
    return_counts=True
)

print(
    f"Уникальных камер до сжатия: "
    f"{len(cats)}"
)

print("\nТоп-10 камер по частоте:")

order = np.argsort(-cat_counts)[:10]

for i in order:

    print(
        f"  cam={int(cats[i]):>3}: "
        f"{cat_counts[i]} "
        f"({cat_counts[i] / n_rows * 100:.2f}%)"
    )

# Редкие категории: менее 1%.

threshold = 0.01 * n_rows

rare = cats[
    cat_counts < threshold
]

print(
    f"\nПорог редкости: {threshold:.2f}"
)

print(
    f"Редких камер: {len(rare)}"
)

# Код OTHER = 255.
# Проверяем, что он не занят существующей камерой.

other_code = 255

if np.any(cats == other_code) and len(rare) > 0:

    # Если 255 уже используется, выбираем 0,
    # если он свободен.

    if not np.any(cats == 0):
        other_code = 0
    else:
        raise ValueError(
            "Коды OTHER=255 и OTHER=0 уже заняты. "
            "Необходимо выбрать свободный код."
        )

mask_rare = np.isin(
    data['cam_id'],
    rare
)

data['cam_id'] = np.where(
    mask_rare,
    np.int16(other_code),
    data['cam_id']
).astype('i2')

cats_after, counts_after = np.unique(
    data['cam_id'],
    return_counts=True
)

print(
    f"\nУникальных категорий после сжатия: "
    f"{len(cats_after)}"
)

print(
    f"Заменено на OTHER={other_code}: "
    f"{mask_rare.sum()} "
    f"({mask_rare.mean() * 100:.2f}%)"
)

print()


# ============================================================
# ФИНАЛЬНОЕ СОХРАНЕНИЕ
# ============================================================

print("=" * 60)
print("ФИНАЛЬНОЕ СОХРАНЕНИЕ")
print("=" * 60)

np.save(
    OUT_BIN,
    data
)

print(f"Файл: {OUT_BIN}")

print(
    f"Количество строк: {data.shape[0]}"
)

print(
    f"Объём памяти: "
    f"{data.nbytes} байт "
    f"= {data.nbytes / 1024 / 1024:.3f} МБ"
)

print("\nПоля итогового массива:")

for name in data.dtype.names:

    print(
        f"  {name:>15}: "
        f"{data.dtype[name]}"
    )

# Проверка сохранённого файла.

check = np.load(OUT_BIN)

print(
    f"\nПроверка загрузки: "
    f"shape={check.shape}"
)

print(
    f"dtype={check.dtype}"
)

print("\nПервая запись:")

print(check[0])
