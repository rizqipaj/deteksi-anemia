"""
anemia_core.py
Modul inti untuk pipeline deteksi anemia dari foto konjungtiva.
Tidak ada logika UI/Streamlit di sini -- supaya bisa ditest terpisah.

Pipeline:
  1. Terima gambar yang SUDAH di-crop manual oleh user (area konjungtiva).
  2. Hapus piksel putih akibat flash kamera (sesuai metode Noor et al.).
  3. Hitung %Red, %Green, %Blue dari piksel yang tersisa (formula Noor et al.).
  4. Encode Sex, susun fitur [sex, %R, %G, %B], scale, lalu prediksi dengan
     model Naive Bayes yang sudah dilatih.
"""

import numpy as np
from PIL import Image
import joblib
import os

# ----------------------------------------------------------------------
# 1. LOAD MODEL & ENCODER (dipanggil sekali saat aplikasi start)
# ----------------------------------------------------------------------

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_artifacts(model_dir: str = MODEL_DIR) -> dict:
    """
    Load semua file .pkl yang dibutuhkan.
    Mengembalikan dict supaya mudah dipanggil di Streamlit dengan @st.cache_resource.
    """
    artifacts = {
        "model": joblib.load(os.path.join(model_dir, "model_naive_bayes.pkl")),
        "scaler": joblib.load(os.path.join(model_dir, "scaler.pkl")),
        "le_sex": joblib.load(os.path.join(model_dir, "label_encoder_sex.pkl")),
        "le_target": joblib.load(os.path.join(model_dir, "label_encoder_target.pkl")),
    }
    return artifacts


# ----------------------------------------------------------------------
# 2. FLASH / WHITE PIXEL REMOVAL (sesuai Noor et al.)
# ----------------------------------------------------------------------

def remove_white_pixels(image_rgb: np.ndarray, threshold: int = 240) -> np.ndarray:
    """
    Menghapus (mask) piksel yang mendekati putih akibat flash kamera.

    Parameters
    ----------
    image_rgb : np.ndarray, shape (H, W, 3), dtype uint8
        Gambar hasil crop manual (RGB, bukan grayscale).
    threshold : int
        Nilai ambang. Piksel dengan R,G,B SEMUA > threshold akan dibuang.
        Default 240 (rentang 0-255) -- sesuai permintaan kamu: nilai tetap.

    Returns
    -------
    pixels_kept : np.ndarray, shape (N, 3)
        Daftar piksel RGB yang TERSISA setelah flash dibuang (bukan gambar,
        tapi array piksel -- karena untuk hitung fitur kita cuma butuh
        daftar piksel, bukan posisi spasialnya).
    mask : np.ndarray, shape (H, W), dtype bool
        Mask boolean (True = piksel dibuang / flash, False = piksel disimpan).
        Dikembalikan juga supaya bisa divisualisasikan di UI nanti.
    """
    if image_rgb.shape[-1] == 4:  # buang alpha channel kalau ada (PNG RGBA)
        image_rgb = image_rgb[:, :, :3]

    r = image_rgb[:, :, 0].astype(int)
    g = image_rgb[:, :, 1].astype(int)
    b = image_rgb[:, :, 2].astype(int)

    is_white_flash = (r > threshold) & (g > threshold) & (b > threshold)

    pixels_kept = image_rgb[~is_white_flash]

    return pixels_kept, is_white_flash


def visualize_masked_image(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Untuk keperluan preview di UI: kembalikan gambar dengan area flash
    ditandai warna magenta terang, supaya user bisa lihat area mana yang dibuang.
    """
    if image_rgb.shape[-1] == 4:
        image_rgb = image_rgb[:, :, :3]
    vis = image_rgb.copy()
    vis[mask] = [255, 0, 255]  # magenta = area yang dibuang
    return vis


# ----------------------------------------------------------------------
# 3. EKSTRAKSI FITUR RGB (formula Noor et al.)
# ----------------------------------------------------------------------

def extract_rgb_percentage(pixels_kept: np.ndarray) -> tuple[float, float, float]:
    """
    Formula Noor et al.:
      %Red   = total_R / (total_R + total_G + total_B) * 100
      %Green = total_G / (total_R + total_G + total_B) * 100
      %Blue  = total_B / (total_R + total_G + total_B) * 100

    Parameters
    ----------
    pixels_kept : np.ndarray, shape (N, 3)
        Output dari remove_white_pixels().

    Returns
    -------
    (pct_red, pct_green, pct_blue) : tuple of float
    """
    if pixels_kept.size == 0:
        raise ValueError(
            "Semua piksel terbuang setelah flash-removal. "
            "Threshold mungkin terlalu rendah, atau area crop terlalu kecil/terang."
        )

    total_r = pixels_kept[:, 0].astype(np.float64).sum()
    total_g = pixels_kept[:, 1].astype(np.float64).sum()
    total_b = pixels_kept[:, 2].astype(np.float64).sum()
    grand_total = total_r + total_g + total_b

    pct_red = (total_r / grand_total) * 100
    pct_green = (total_g / grand_total) * 100
    pct_blue = (total_b / grand_total) * 100

    return pct_red, pct_green, pct_blue


# ----------------------------------------------------------------------
# 4. PREDIKSI
# ----------------------------------------------------------------------

def predict_anemia(
    artifacts: dict,
    sex: str,
    cropped_image_rgb: np.ndarray,
    white_threshold: int = 240,
) -> dict:
    """
    Pipeline penuh: dari gambar crop manual -> hasil prediksi.

    Parameters
    ----------
    artifacts : dict
        Hasil dari load_artifacts().
    sex : str
        "M" atau "F" (harus sesuai classes_ di label_encoder_sex.pkl).
    cropped_image_rgb : np.ndarray
        Gambar HASIL CROP MANUAL user (sudah berupa area konjungtiva saja).
    white_threshold : int
        Threshold flash removal.

    Returns
    -------
    dict berisi:
        - pct_red, pct_green, pct_blue : fitur yang dihitung
        - sex_encoded : hasil encoding sex
        - prediction_label : "Yes" / "No" (hasil decode)
        - prediction_proba : dict {"No": ..., "Yes": ...} confidence score
        - mask : array boolean area flash (untuk visualisasi)
    """
    pixels_kept, mask = remove_white_pixels(cropped_image_rgb, threshold=white_threshold)
    pct_red, pct_green, pct_blue = extract_rgb_percentage(pixels_kept)

    sex_encoded = artifacts["le_sex"].transform([sex])[0]

    # URUTAN FITUR HARUS SAMA DENGAN SAAT TRAINING:
    # [sex, %Red, %Green, %Blue]  <-- sesuai urutan mean_ di scaler.pkl
    feature_vector = np.array([[sex_encoded, pct_red, pct_green, pct_blue]])

    feature_scaled = artifacts["scaler"].transform(feature_vector)

    pred_encoded = artifacts["model"].predict(feature_scaled)[0]
    pred_label = artifacts["le_target"].inverse_transform([pred_encoded])[0]

    proba = artifacts["model"].predict_proba(feature_scaled)[0]
    proba_dict = {
        cls: float(p)
        for cls, p in zip(artifacts["le_target"].classes_, proba)
    }

    return {
        "pct_red": pct_red,
        "pct_green": pct_green,
        "pct_blue": pct_blue,
        "sex_encoded": int(sex_encoded),
        "prediction_label": pred_label,
        "prediction_proba": proba_dict,
        "mask": mask,
    }


def validate_selection(n_pixels_total: int, n_pixels_flash: int, min_pixels: int = 500) -> list[str]:
    """
    Tahap 4: cek kewajaran area yang dipilih user sebelum/​setelah prediksi.
    Mengembalikan list pesan peringatan (string). List kosong = tidak ada masalah.

    Parameters
    ----------
    n_pixels_total : int
        Jumlah piksel yang dicat user (sebelum flash-removal).
    n_pixels_flash : int
        Jumlah piksel di antaranya yang dibuang karena terdeteksi flash putih.
    min_pixels : int
        Ambang minimum piksel valid (setelah flash dibuang) agar dianggap
        cukup representatif. Default 500 -- nilai aman untuk foto 12MP yang
        diresize ke lebar canvas 500px.
    """
    warnings = []
    n_valid = n_pixels_total - n_pixels_flash

    if n_valid < min_pixels:
        warnings.append(
            f"Area yang dipilih terlalu kecil ({n_valid} piksel valid setelah flash "
            f"dibuang, minimal disarankan {min_pixels}). Hasil mungkin kurang akurat -- "
            f"coba cat area konjungtiva yang lebih luas."
        )

    if n_pixels_total > 0:
        flash_ratio = n_pixels_flash / n_pixels_total
        if flash_ratio > 0.5:
            warnings.append(
                f"Lebih dari separuh area yang dicat ({flash_ratio:.0%}) terdeteksi "
                f"sebagai highlight flash. Kemungkinan area yang dipilih kebanyakan "
                f"kena pantulan cahaya, bukan jaringan konjungtiva -- coba pindah ke "
                f"area yang tidak terlalu mengkilap."
            )

    return warnings


def get_selection_bbox(mask_full: np.ndarray, padding: int = 20) -> tuple[int, int, int, int]:
    """
    Tahap 4: hitung bounding box (kotak pembatas) dari area yang dicat user,
    dengan padding di sekelilingnya, supaya bisa dibuat preview "zoom" ke
    area yang dianalisis (bukan menampilkan foto penuh).

    Parameters
    ----------
    mask_full : np.ndarray, shape (H, W), dtype bool
        Mask brush user pada resolusi gambar ASLI (bukan resolusi canvas).
    padding : int
        Jumlah piksel tambahan di sekeliling bounding box, supaya area
        sekitar konjungtiva (untuk konteks visual) tetap sedikit terlihat.

    Returns
    -------
    (top, bottom, left, right) : tuple of int
        Koordinat untuk slicing: image[top:bottom, left:right].
    """
    rows = np.any(mask_full, axis=1)
    cols = np.any(mask_full, axis=0)
    if not rows.any() or not cols.any():
        h, w = mask_full.shape
        return 0, h, 0, w

    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    h, w = mask_full.shape
    top = max(0, int(top) - padding)
    bottom = min(h, int(bottom) + padding + 1)
    left = max(0, int(left) - padding)
    right = min(w, int(right) + padding + 1)

    return top, bottom, left, right


def predict_anemia_from_pixels(
    artifacts: dict,
    sex: str,
    selected_pixels_rgb: np.ndarray,
    white_threshold: int = 240,
) -> dict:
    """
    Versi predict_anemia() yang menerima DAFTAR PIKSEL secara langsung,
    bukan gambar persegi (rectangle). Dipakai untuk Tahap 3 (crop
    freeform/brush) -- piksel yang masuk ke sini adalah piksel yang
    SUDAH DIPILIH user lewat brush di canvas, jadi tidak perlu di-crop
    rectangle lagi.

    Tetap melalui tahap flash-removal yang sama (highlight putih akibat
    flash bisa saja tetap tercoret brush oleh user secara tidak sengaja).

    Parameters
    ----------
    artifacts : dict
        Hasil dari load_artifacts().
    sex : str
        "M" atau "F".
    selected_pixels_rgb : np.ndarray, shape (N, 3)
        Piksel-piksel RGB yang dipilih user lewat brush (sudah flat,
        bukan gambar 2D).
    white_threshold : int
        Threshold flash removal.

    Returns
    -------
    dict dengan struktur sama seperti predict_anemia(), kecuali key
    "mask" diganti dengan "n_pixels_total" dan "n_pixels_flash" (karena
    di sini tidak ada lagi struktur gambar 2D untuk divisualisasikan
    sebagai mask spasial).
    """
    if selected_pixels_rgb.size == 0:
        raise ValueError(
            "Tidak ada piksel yang dipilih. Pastikan area konjungtiva "
            "sudah dicat dengan brush sebelum menekan tombol Prediksi."
        )

    r = selected_pixels_rgb[:, 0].astype(int)
    g = selected_pixels_rgb[:, 1].astype(int)
    b = selected_pixels_rgb[:, 2].astype(int)
    is_white_flash = (r > white_threshold) & (g > white_threshold) & (b > white_threshold)

    pixels_kept = selected_pixels_rgb[~is_white_flash]

    pct_red, pct_green, pct_blue = extract_rgb_percentage(pixels_kept)

    sex_encoded = artifacts["le_sex"].transform([sex])[0]
    feature_vector = np.array([[sex_encoded, pct_red, pct_green, pct_blue]])
    feature_scaled = artifacts["scaler"].transform(feature_vector)

    pred_encoded = artifacts["model"].predict(feature_scaled)[0]
    pred_label = artifacts["le_target"].inverse_transform([pred_encoded])[0]

    proba = artifacts["model"].predict_proba(feature_scaled)[0]
    proba_dict = {
        cls: float(p)
        for cls, p in zip(artifacts["le_target"].classes_, proba)
    }

    return {
        "pct_red": pct_red,
        "pct_green": pct_green,
        "pct_blue": pct_blue,
        "sex_encoded": int(sex_encoded),
        "prediction_label": pred_label,
        "prediction_proba": proba_dict,
        "n_pixels_total": int(selected_pixels_rgb.shape[0]),
        "n_pixels_flash": int(is_white_flash.sum()),
    }
