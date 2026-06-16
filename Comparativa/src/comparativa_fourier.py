"""
comparativa_fourier.py
----------------------
Script de comparación académica para series de Fourier.
Lee su configuración desde 'config.yaml' en la raíz del proyecto.

Uso:
    pixi run python src/comparativa_fourier.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import yaml
from pathlib import Path

# ============================================================================
# CARGAR CONFIGURACIÓN
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# Rutas de datos y salida
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "figuras"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# APLICAR ESTILO GLOBAL
# ============================================================================
out_cfg = CFG["output"]
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": out_cfg["dpi"],
    "savefig.format": "png",
    "savefig.bbox": "tight",
    "font.family": out_cfg["font_family"],
    "font.serif": [out_cfg["font_serif"], "DejaVu Serif", "serif"],
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.facecolor": out_cfg["facecolor"],
    "figure.facecolor": out_cfg["facecolor"],
    "axes.edgecolor": out_cfg["edgecolor"],
    "axes.linewidth": out_cfg["axes_linewidth"],
    "xtick.direction": "in",
    "ytick.direction": "in",
    "ytick.right": True,
    "xtick.top": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
})

# ============================================================================
# 1. LEER DATOS (SOLO CSVs — sin Excel)
# ============================================================================
# Los CSVs de hilos/procesos/MPI comparten exactamente la misma grilla de x.
# Por tanto NO se requiere merge outer ni interpolacion de ningun tipo.
# El unico merge necesario es unir las columnas F(X) de cada CSV sobre el
# mismo indice x (inner join), lo cual preserva todos los puntos originales.

csv_files = sorted(glob.glob(str(DATA_DIR / "Con_*.csv")))
print(f"[INFO] CSVs detectados: {csv_files}")
if not csv_files:
    raise FileNotFoundError("No se encontraron archivos Con_*.csv en data/")

# Usar el primer CSV como esqueleto base (columna x + primera aproximacion)
first_csv = csv_files[0]
df = pd.read_csv(first_csv, skiprows=3, header=0)
df = df[["x", "F(X)"]].rename(columns={"F(X)": f"F_{os.path.splitext(os.path.basename(first_csv))[0]}"})

# Anexar los demas CSVs (misma grilla, misma longitud, sin NaN)
for csv_path in csv_files[1:]:
    base = os.path.splitext(os.path.basename(csv_path))[0]
    df_csv = pd.read_csv(csv_path, skiprows=3, header=0)
    df_csv = df_csv[["x", "F(X)"]].rename(columns={"F(X)": f"F_{base}"})
    # merge inner: conserva solo los x presentes en ambos (en este caso, todos)
    df = df.merge(df_csv, on="x", how="inner")

df = df.sort_values("x").reset_index(drop=True)

# ============================================================================
# FUNCION REAL: f(x) = x^4 - 3x  (recalculada directamente, exacta)
# ============================================================================
# La funcion real es conocida analiticamente. En lugar de leerla de un Excel
# con una grilla potencialmente diferente, la recalculamos directamente sobre
# los x de los CSVs. Esto garantiza:
#   - Exactitud matematica en cada punto
#   - Alineacion perfecta con las aproximaciones
#   - Cero dependencia de archivos externos
# ============================================================================
df["f_real"] = df["x"].values**4 - 3*df["x"].values

# Como todos los CSVs comparten la misma grilla exacta, no hay NaN en las
# columnas de metodos. No se requiere interpolacion.
# df_orig ya no es necesario; df es fiel a los datos originales.

# Columnas de métodos
method_cols = [c for c in df.columns if c.startswith("F_")]
method_labels = {}
for c in method_cols:
    method_labels[c] = c.replace("F_Con_", "").replace("_", " ").title()

# Ordenar métodos según config
order_cfg = CFG.get("method_order")
if order_cfg:
    # Construir mapa inverso label -> col_name
    label_to_col = {v: k for k, v in method_labels.items()}
    ordered_cols = []
    for lbl in order_cfg:
        if lbl in label_to_col:
            ordered_cols.append(label_to_col[lbl])
    # Añadir los que no estén en la lista al final
    for c in method_cols:
        if c not in ordered_cols:
            ordered_cols.append(c)
    method_cols = ordered_cols

x = df["x"].values
f_real = df["f_real"].values

# ============================================================================
# UTILIDADES DE ESTILO (desde config)
# ============================================================================
PAL = CFG["palette"]
REAL_COLOR = PAL["real"]
METHOD_COLORS = PAL["methods"]

LINE_CFG = CFG["lines"]
REAL_STYLE = LINE_CFG["real"]["style"]
REAL_WIDTH = LINE_CFG["real"]["width"]
REAL_ZORDER = LINE_CFG["real"]["zorder"]
METHOD_STYLES = [tuple(s) if isinstance(s, list) else s for s in LINE_CFG["methods"]["styles"]]
METHOD_WIDTH = LINE_CFG["methods"]["width"]

def method_color(i):
    return METHOD_COLORS[i % len(METHOD_COLORS)]

def method_linestyle(i):
    return METHOD_STYLES[i % len(METHOD_STYLES)]

GLOW_CFG = CFG["glow"]

def plot_glow(ax, xdata, ydata, color, zorder_base=1):
    """Dibuja capas de halo degradado detrás de la curva."""
    if not GLOW_CFG["enabled"]:
        return
    for layer in GLOW_CFG["layers"]:
        ax.plot(xdata, ydata, color=color, linewidth=layer["width"], alpha=layer["alpha"],
                solid_capstyle="round", zorder=zorder_base)


def apply_title(ax, title_cfg):
    """Aplica titulo con fontsize, loc y pad desde config."""
    kw = {"fontsize": title_cfg["fontsize"]} if "fontsize" in title_cfg else {}
    if "loc" in title_cfg:
        kw["loc"] = title_cfg["loc"]
    if "pad" in title_cfg:
        kw["pad"] = title_cfg["pad"]
    ax.set_title(title_cfg["text"], **kw)


def apply_xlabel(ax, label_cfg):
    """Aplica etiqueta X con fontsize desde config."""
    kw = {"fontsize": label_cfg["fontsize"]} if "fontsize" in label_cfg else {}
    ax.set_xlabel(label_cfg["text"], **kw)


def apply_ylabel(ax, label_cfg):
    """Aplica etiqueta Y con fontsize desde config."""
    kw = {"fontsize": label_cfg["fontsize"]} if "fontsize" in label_cfg else {}
    ax.set_ylabel(label_cfg["text"], **kw)

# ============================================================================
# CORTES / SEGMENTACIÓN
# ============================================================================
cuts_cfg = CFG["cuts"]["real"]

def segment_real(xarr, yarr, n_cuts):
    """Divide el array en n_cuts segmentos iguales insertando NaN entre ellos."""
    if n_cuts <= 1:
        return xarr, yarr
    n = len(xarr)
    step = n // n_cuts
    xs, ys = [], []
    for i in range(n_cuts):
        start = i * step
        end = (i + 1) * step if i < n_cuts - 1 else n
        xs.extend(xarr[start:end])
        ys.extend(yarr[start:end])
        if i < n_cuts - 1:
            xs.append(np.nan)
            ys.append(np.nan)
    return np.array(xs), np.array(ys)

x_real_seg, f_real_seg = segment_real(x, f_real, cuts_cfg)

# ============================================================================
# 2. MÉTRICAS (sobre datos originales, sin interpolacion)
# ============================================================================
# Nota: ya que todos los CSVs comparten la misma grilla exacta, df no contiene
# NaN en las columnas de metodos. Las metricas se calculan punto a punto real.
metrics = []
for col in method_cols:
    diff = df[col].values - df["f_real"].values
    mae  = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    max_err = np.max(np.abs(diff))
    metrics.append({
        "Método": method_labels[col],
        "MAE": mae,
        "RMSE": rmse,
        "Máx": max_err,
    })
df_metrics = pd.DataFrame(metrics)

# ============================================================================
# 3. GRAFICAS
# ============================================================================

# --- Fig 01: Superposición -------------------------------------------------
c01 = CFG["fig01_superposicion"]
if c01["enabled"]:
    fig, ax = plt.subplots(figsize=c01["figsize"])
    ax.plot(x_real_seg, f_real_seg, label="$f(x)$ real", color=REAL_COLOR,
            linewidth=REAL_WIDTH, linestyle=REAL_STYLE, zorder=REAL_ZORDER)
    for i, col in enumerate(method_cols):
        y = df[col].values
        color = method_color(i)
        plot_glow(ax, x, y, color, zorder_base=GLOW_CFG["zorder_base"] + i * 3)
        ax.plot(x, y, label=method_labels[col], color=color,
                linewidth=METHOD_WIDTH, linestyle=method_linestyle(i),
                zorder=5 + i)
    apply_xlabel(ax, c01["xlabel"])
    apply_ylabel(ax, c01["ylabel"])
    apply_title(ax, c01["title"])
    ax.legend(loc=c01["legend"]["loc"], ncol=c01["legend"]["ncol"])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_superposicion.png")
    plt.close(fig)

# --- Fig 02: Error absoluto ------------------------------------------------
c02 = CFG["fig02_error_absoluto"]
if c02["enabled"]:
    fig, ax = plt.subplots(figsize=c02["figsize"])
    for i, col in enumerate(method_cols):
        err = np.abs(df[col].values - f_real)
        color = method_color(i)
        plot_glow(ax, x, err, color, zorder_base=GLOW_CFG["zorder_base"] + i * 3)
        ax.plot(x, err, label=method_labels[col], color=color,
                linewidth=METHOD_WIDTH, linestyle=method_linestyle(i),
                zorder=5 + i)
    apply_xlabel(ax, c02["xlabel"])
    apply_ylabel(ax, c02["ylabel"])
    apply_title(ax, c02["title"])
    ax.set_yscale(c02["yscale"])
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_errores_absolutos.png")
    plt.close(fig)

# --- Fig 03: Métricas de error (barras) ------------------------------------
c03 = CFG["fig03_metricas"]
if c03["enabled"]:
    df_metrics.to_csv(OUT_DIR / "metricas_error.csv", index=False, float_format="%.6e")
    fig, axes = plt.subplots(1, 3, figsize=c03["figsize"])
    x_pos = np.arange(len(df_metrics))
    width = c03["bar_width"]
    bar_colors_cfg = c03.get("bar_colors")
    if bar_colors_cfg:
        bar_colors = bar_colors_cfg
    else:
        bar_colors = [method_color(i) for i in range(len(df_metrics))]
    edge_cfg = c03["bar_edge"]
    t_cfg = c03.get("title_per_subplot", {})
    for ax, col_name in zip(axes, c03["metrics"]):
        bars = ax.bar(x_pos, df_metrics[col_name], width, color=bar_colors,
                      edgecolor=edge_cfg["color"], linewidth=edge_cfg["width"])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_metrics["Método"], rotation=30, ha="right")
        ax.set_ylabel(col_name)
        ax.set_title(col_name, fontsize=t_cfg.get("fontsize"),
                     loc=t_cfg.get("loc"), pad=t_cfg.get("pad"))
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                txt_cfg = c03["text"]
                ypos = h / 2 if txt_cfg["position"] == "center" else h
                va = "center" if txt_cfg["position"] == "center" else "bottom"
                ax.annotate(
                    f"{h:.2e}",
                    xy=(bar.get_x() + bar.get_width() / 2, ypos),
                    ha="center", va=va,
                    fontsize=txt_cfg["fontsize"],
                    color=txt_cfg["color"],
                    fontweight=txt_cfg["fontweight"]
                )
    s_cfg = c03["suptitle"]
    fig.suptitle(s_cfg["text"], fontsize=s_cfg.get("fontsize"), y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_metricas_error.png")
    plt.close(fig)

# --- Fig 04: Fenómeno de Gibbs ---------------------------------------------
c04 = CFG["fig04_gibbs"]
if c04["enabled"]:
    zooms = c04["zooms"]
    fig, axes = plt.subplots(1, len(zooms), figsize=c04["figsize"], sharey=True)
    if len(zooms) == 1:
        axes = [axes]
    for ax, (xmin, xmax) in zip(axes, zooms):
        mask = (x >= xmin) & (x <= xmax)
        ax.plot(x_real_seg[mask], f_real_seg[mask], label="$f(x)$ real",
                color=REAL_COLOR, linewidth=REAL_WIDTH, linestyle=REAL_STYLE,
                zorder=REAL_ZORDER)
        for i, col in enumerate(method_cols):
            y = df[col].values[mask]
            color = method_color(i)
            plot_glow(ax, x[mask], y, color, zorder_base=GLOW_CFG["zorder_base"] + i * 3)
            ax.plot(x[mask], y, label=method_labels[col], color=color,
                    linewidth=METHOD_WIDTH, linestyle=method_linestyle(i),
                    zorder=5 + i)
        apply_xlabel(ax, c04["xlabel"])
        t_cfg = c04["title"]
        title_text = f"{t_cfg['text']} $[{xmin:.2f},\\; {xmax:.2f}]$"
        ax.set_title(title_text, fontsize=t_cfg.get("fontsize"),
                     loc=t_cfg.get("loc"), pad=t_cfg.get("pad"))
        ax.legend(loc="best", fontsize=9)
    apply_ylabel(axes[0], c04["ylabel"])
    s_cfg = c04["suptitle"]
    fig.suptitle(s_cfg["text"], fontsize=s_cfg.get("fontsize"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_fenomeno_gibbs.png")
    plt.close(fig)

# --- Fig 05: Error cuadrático ----------------------------------------------
c05 = CFG["fig05_error_cuadratico"]
if c05["enabled"]:
    fig, ax = plt.subplots(figsize=c05["figsize"])
    for i, col in enumerate(method_cols):
        err2 = (df[col].values - f_real) ** 2
        ax.plot(x, err2, label=method_labels[col],
                color=method_color(i), linewidth=METHOD_WIDTH,
                linestyle=method_linestyle(i))
    apply_xlabel(ax, c05["xlabel"])
    apply_ylabel(ax, c05["ylabel"])
    apply_title(ax, c05["title"])
    ax.set_yscale(c05["yscale"])
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_errores_cuadraticos.png")
    plt.close(fig)

# --- Fig 06: Paneles individuales ------------------------------------------
c06 = CFG["fig06_paneles"]
if c06["enabled"]:
    n_methods = len(method_cols)
    cols = c06["cols"]
    rows = int(np.ceil(n_methods / cols))
    fw, fh = c06["figsize_per_panel"]
    fig, axes = plt.subplots(rows, cols, figsize=(fw * cols, fh * rows), squeeze=False)
    for idx, col in enumerate(method_cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.plot(x_real_seg, f_real_seg, label="$f(x)$ real", color=REAL_COLOR,
                linewidth=REAL_WIDTH, linestyle=REAL_STYLE, zorder=REAL_ZORDER)
        y = df[col].values
        color = method_color(idx)
        plot_glow(ax, x, y, color, zorder_base=4)
        ax.plot(x, y, label=method_labels[col], color=color,
                linewidth=METHOD_WIDTH, linestyle=method_linestyle(idx), zorder=5)
        ax.fill_between(x, f_real, y, color=color, alpha=c06["fill_alpha"])
        apply_xlabel(ax, c06["xlabel"])
        apply_ylabel(ax, c06["ylabel"])
        t_cfg = c06.get("title", {})
        ax.set_title(method_labels[col], fontsize=t_cfg.get("fontsize"),
                     loc=t_cfg.get("loc"), pad=t_cfg.get("pad"))
        ax.legend(loc="best", fontsize=9)
    for idx in range(n_methods, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
    s_cfg = c06["suptitle"]
    fig.suptitle(s_cfg["text"], fontsize=s_cfg.get("fontsize"), y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_paneles_individuales.png")
    plt.close(fig)

# ============================================================================
# 4. REPORTE MARKDOWN
# ============================================================================
rpt = CFG["reporte"]
if rpt["enabled"]:
    md_path = OUT_DIR / rpt["filename"]
    float_fmt = rpt["float_format"]
    md_table = df_metrics.to_markdown(index=False, floatfmt=float_fmt) if rpt["include_table"] else ""

    reporte_content = f"""# Reporte de Comparativa – Series de Fourier

## 1. Superposicion global

![01_superposicion](01_superposicion.png)

**Figura 1.** Comparación global de la función real $f(x)=x^4-3x$ (línea negra sólida) con las aproximaciones de Fourier calculadas mediante cómputo paralelo (hilos, procesos y MPI). Cada aproximación se dibuja con un patrón discontinuo y un color distintivo de la paleta "caos", superponiéndose sin cortes visuales. El halo degradado (*glow*) de cada curva se sitúa en el fondo para suavizar las transiciones.

---

## 2. Error absoluto punto a punto

![02_errores_absolutos](02_errores_absolutos.png)

**Figura 2.** Error absoluto $|F(x)-f(x)|$ en escala logarítmica. Valores menores indican mayor precisión de la aproximación. Los picos recurrentes evidencian las oscilaciones propias del fenómeno de Gibbs en las proximidades de las discontinuidades periódicas.

---

## 3. Métricas de error

![03_metricas_error](03_metricas_error.png)

**Figura 3.** Barras comparativas del MAE, RMSE y error maximo absoluto para cada metodo de calculo.

### Tabla resumen

{md_table}

**Tabla 1.** Resumen numerico de las metricas de error para cada metodo evaluado.

---

## 4. Fenomeno de Gibbs

![04_fenomeno_gibbs](04_fenomeno_gibbs.png)

**Figura 4.** Ampliación de las regiones adyacentes a las discontinuidades periódicas ($\\pm\\pi$). El *overshoot* característico del fenómeno de Gibbs es observable en el extremo derecho de cada panel.

---

## 5. Error cuadratico

![05_errores_cuadraticos](05_errores_cuadraticos.png)

**Figura 5.** Error cuadrático $(F(x)-f(x))^2$ en escala logarítmica. Esta métrica amplifica las desviaciones de mayor magnitud.

---

## 6. Comparación individual por paneles

![06_paneles_individuales](06_paneles_individuales.png)

**Figura 6.** Paneles individuales que contrastan cada aproximación (línea discontinua) con la función real (negro sólido). El sombreado semitransparente resalta la zona de discrepancia.

---

## Conclusiones

- **Equivalencia numérica:** Los métodos implementados en C (hilos, procesos y MPI) producen resultados idénticos en precisión.
- **Fenómeno de Gibbs:** Todas las aproximaciones exhiben el clásico *overshoot* en las discontinuidades de $\\pm\\pi$.
- **Exactitud de la línea base:** La función real $f(x)=x^4-3x$ se recalcula directamente sobre la grilla de cada CSV, eliminando toda dependencia de grillas externas y garantizando métricas de error fidedignas.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(reporte_content)
    print(f"   Reporte generado:     {md_path.resolve()}")

print("\n✅ Comparativa finalizada.")
print(f"   Figuras guardadas en: {OUT_DIR.resolve()}")
print("\nMétricas de error:")
print(df_metrics.to_string(index=False))
