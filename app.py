"""
app.py
TAHAP 4 (REVISI) -- Dashboard dengan tema custom, alur step-by-step,
input nama pengguna, panduan foto benar/salah, dan grafik confidence
interaktif (Plotly).

Perubahan dibanding revisi sebelumnya:
1. Header & tema dipercantik (CSS custom, warna konsisten).
2. Alur dipecah jadi langkah bernomor (1. Identitas -> 2. Panduan foto
   -> 3. Upload & crop -> 4. Hasil) supaya lebih jelas urutannya.
3. Input Nama ditambahkan di awal (dipakai untuk personalisasi &
   laporan unduhan).
4. Panduan foto benar vs salah ditampilkan sebelum upload, memakai
   2 gambar contoh (Contoh_Foto_Benar.jpeg, Contoh_Foto_Salah.jpeg).
5. Confidence score ditampilkan sebagai gauge chart interaktif
   (Plotly) -- bukan cuma teks/metric.

Logika inti (flash-removal, ekstraksi fitur, prediksi, brush masking,
validasi, bounding box) TIDAK diubah dari Tahap 4 sebelumnya.
"""

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import plotly.graph_objects as go

from streamlit_drawable_canvas import st_canvas

from anemia_core import (
    load_artifacts,
    predict_anemia_from_pixels,
    validate_selection,
    get_selection_bbox,
)

# ----------------------------------------------------------------------
# KONFIGURASI HALAMAN & TEMA
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Anemia - Konjungtiva",
    page_icon="🩸",
    layout="centered",
)

PRIMARY = "#C0392B"
ACCENT = "#1A5276"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #FAFAFA;
    }}
    .anemia-header {{
        background: linear-gradient(135deg, {PRIMARY} 0%, #922B21 100%);
        padding: 28px 24px;
        border-radius: 14px;
        margin-bottom: 22px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }}
    .anemia-header h1 {{
        margin: 0;
        font-size: 1.9rem;
    }}
    .anemia-header p {{
        margin: 6px 0 0 0;
        font-size: 0.95rem;
        opacity: 0.92;
    }}
    .step-badge {{
        display: inline-block;
        background-color: {PRIMARY};
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        text-align: center;
        line-height: 28px;
        font-weight: 700;
        margin-right: 10px;
        font-size: 0.95rem;
    }}
    .step-title {{
        font-size: 1.2rem;
        font-weight: 700;
        color: {ACCENT};
        margin-bottom: 4px;
    }}
    .guide-card {{
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        font-weight: 600;
    }}
    .guide-good {{
        background-color: #EAFAF1;
        border: 1px solid #27AE60;
        color: #1E8449;
    }}
    .guide-bad {{
        background-color: #FDEDEC;
        border: 1px solid #E74C3C;
        color: #C0392B;
    }}
    </style>

    <div class="anemia-header">
        <h1>🩸 Deteksi Anemia dari Foto Konjungtiva</h1>
        <p>Berbasis analisis warna konjungtiva mata &amp; model statistik Naive Bayes</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def step_header(number: int, title: str):
    st.markdown(
        f'<div><span class="step-badge">{number}</span>'
        f'<span class="step-title">{title}</span></div>',
        unsafe_allow_html=True,
    )


WHITE_THRESHOLD = 240
CANVAS_WIDTH = 500
MIN_VALID_PIXELS = 500

# ----------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------
@st.cache_resource
def get_artifacts():
    return load_artifacts()


artifacts = get_artifacts()

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


def add_to_history(entry: dict):
    st.session_state.history.insert(0, entry)


def build_report_text(entry: dict) -> str:
    lines = [
        "=" * 50,
        "LAPORAN HASIL DETEKSI ANEMIA - KONJUNGTIVA",
        "=" * 50,
        f"Nama           : {entry['name']}",
        f"Waktu          : {entry['timestamp']}",
        f"Nama file foto : {entry['filename']}",
        f"Jenis kelamin  : {entry['sex']}",
        "-" * 50,
        f"%Red           : {entry['pct_red']:.2f}%",
        f"%Green         : {entry['pct_green']:.2f}%",
        f"%Blue          : {entry['pct_blue']:.2f}%",
        "-" * 50,
        f"Piksel dipilih (brush) : {entry['n_pixels_total']}",
        f"Piksel dibuang (flash) : {entry['n_pixels_flash']}",
        f"Piksel valid dipakai   : {entry['n_pixels_total'] - entry['n_pixels_flash']}",
        "-" * 50,
        f"HASIL PREDIKSI : {entry['prediction_label_display']}",
        f"Confidence     : {entry['confidence']:.1%}",
        "-" * 50,
        "Detail confidence per kelas:",
    ]
    for cls, p in entry["prediction_proba"].items():
        lines.append(f"  - {cls}: {p:.1%}")

    if entry["warnings"]:
        lines.append("-" * 50)
        lines.append("PERINGATAN:")
        for w in entry["warnings"]:
            lines.append(f"  - {w}")

    lines.append("=" * 50)
    lines.append(
        "Catatan: Hasil ini dihasilkan oleh sistem prediksi berbasis "
        "model statistik (Naive Bayes) dari fitur warna RGB konjungtiva, "
        "BUKAN diagnosis medis resmi. Konsultasikan dengan tenaga medis "
        "untuk diagnosis yang valid."
    )
    return "\n".join(lines)


def make_gauge(confidence: float, label_display: str) -> go.Figure:
    color = PRIMARY if label_display == "Anemic" else "#27AE60"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": f"Confidence — {label_display}", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#F4F6F6"},
                    {"range": [50, 75], "color": "#FDEBD0"},
                    {"range": [75, 100], "color": "#FADBD8" if label_display == "Anemic" else "#D5F5E3"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": confidence * 100,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def make_rgb_bar(pct_red: float, pct_green: float, pct_blue: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["%Red", "%Green", "%Blue"],
                y=[pct_red, pct_green, pct_blue],
                marker_color=["#E74C3C", "#27AE60", "#2980B9"],
                text=[f"{v:.2f}%" for v in [pct_red, pct_green, pct_blue]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title="Persentase (%)",
        showlegend=False,
    )
    return fig


# ----------------------------------------------------------------------
# LANGKAH 1 -- IDENTITAS PENGGUNA
# ----------------------------------------------------------------------
step_header(1, "Masukkan Data Diri")

col1, col2 = st.columns(2)
with col1:
    user_name = st.text_input("Nama", placeholder="Masukkan nama kamu")
with col2:
    sex = st.radio("Jenis kelamin", options=["M", "F"], horizontal=True)

if not user_name.strip():
    st.info("⬆️ Masukkan nama terlebih dahulu untuk melanjutkan.")
    st.stop()

st.divider()

# ----------------------------------------------------------------------
# LANGKAH 2 -- PANDUAN FOTO
# ----------------------------------------------------------------------
step_header(2, "Panduan Mengambil Foto")
st.write(
    "Tarik kelopak mata bagian bawah dengan jari agar konjungtiva (jaringan "
    "merah muda di bagian dalam kelopak) terlihat jelas, lalu foto **dengan "
    "zoom dekat** seperti contoh berikut:"
)

gcol1, gcol2 = st.columns(2)
with gcol1:
    st.image("Contoh_Foto_Benar.jpeg", use_container_width=True)
    st.markdown('<div class="guide-card guide-good">✅ Contoh Benar — konjungtiva terlihat jelas &amp; close-up</div>', unsafe_allow_html=True)
with gcol2:
    st.image("Contoh_Foto_Salah.jpeg", use_container_width=True)
    st.markdown('<div class="guide-card guide-bad">❌ Contoh Salah — foto wajah penuh, konjungtiva tidak terlihat</div>', unsafe_allow_html=True)

st.caption(
    "💡 Pastikan cahaya cukup terang (tapi tanpa flash langsung ke mata), "
    "dan tidak buram/blur."
)

st.divider()

# ----------------------------------------------------------------------
# LANGKAH 3 -- UPLOAD & CROP
# ----------------------------------------------------------------------
step_header(3, "Upload Foto & Tandai Area Konjungtiva")

uploaded_file = st.file_uploader(
    "Upload foto konjungtiva (sudah zoom ke mata)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload foto sesuai panduan di atas untuk melanjutkan.")
    if st.session_state.history:
        st.divider()
        st.subheader("📜 Riwayat sesi ini")
        for i, entry in enumerate(st.session_state.history):
            with st.expander(
                f"{entry['timestamp']} — {entry['name']} — {entry['filename']} — "
                f"{entry['prediction_label_display']}"
            ):
                st.write(
                    f"%Red {entry['pct_red']:.2f}% | "
                    f"%Green {entry['pct_green']:.2f}% | "
                    f"%Blue {entry['pct_blue']:.2f}%"
                )
                st.download_button(
                    "⬇️ Download laporan ini",
                    data=build_report_text(entry),
                    file_name=f"laporan_anemia_{i}.txt",
                    mime="text/plain",
                    key=f"dl_empty_{i}",
                )
    st.stop()

original_image = Image.open(uploaded_file).convert("RGB")
orig_w, orig_h = original_image.size
scale = CANVAS_WIDTH / orig_w
canvas_height = int(orig_h * scale)

display_image = original_image.resize((CANVAS_WIDTH, canvas_height))

st.write("**Cat (brush) area konjungtiva murni di bawah ini:**")

brush_size = st.slider(
    "Ukuran brush",
    min_value=3,
    max_value=40,
    value=15,
    help="Perbesar brush kalau area konjungtiva cukup luas, "
         "perkecil untuk hasil yang lebih presisi di tepi.",
)

canvas_result = st_canvas(
    fill_color="rgba(0, 255, 0, 0.3)",
    stroke_width=brush_size,
    stroke_color="rgba(0, 255, 0, 0.7)",
    background_image=display_image,
    update_streamlit=True,
    height=canvas_height,
    width=CANVAS_WIDTH,
    drawing_mode="freedraw",
    key="canvas_crop",
)

predict_clicked = st.button("🔍 Prediksi", type="primary")

st.divider()

# ----------------------------------------------------------------------
# LANGKAH 4 -- HASIL
# ----------------------------------------------------------------------
if predict_clicked:
    if canvas_result.image_data is None:
        st.warning("Belum ada area yang dicat. Silakan brush area konjungtiva dulu.")
        st.stop()

    canvas_rgba = canvas_result.image_data.astype(np.uint8)
    brush_mask_small = canvas_rgba[:, :, 3] > 0

    if brush_mask_small.sum() == 0:
        st.warning("Belum ada area yang dicat. Silakan brush area konjungtiva dulu.")
        st.stop()

    mask_image_small = Image.fromarray((brush_mask_small * 255).astype(np.uint8))
    mask_image_full = mask_image_small.resize((orig_w, orig_h), resample=Image.NEAREST)
    brush_mask_full = np.array(mask_image_full) > 127

    original_np = np.array(original_image)
    selected_pixels = original_np[brush_mask_full]

    try:
        with st.spinner("Memproses..."):
            result = predict_anemia_from_pixels(
                artifacts=artifacts,
                sex=sex,
                selected_pixels_rgb=selected_pixels,
                white_threshold=WHITE_THRESHOLD,
            )
            warnings = validate_selection(
                n_pixels_total=result["n_pixels_total"],
                n_pixels_flash=result["n_pixels_flash"],
                min_pixels=MIN_VALID_PIXELS,
            )

        step_header(4, f"Hasil untuk {user_name}")

        for w in warnings:
            st.warning(f"⚠️ {w}")

        overlay = original_np.copy()
        overlay[brush_mask_full] = (
            overlay[brush_mask_full] * 0.5 + np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)

        top, bottom, left, right = get_selection_bbox(brush_mask_full, padding=25)
        zoomed_overlay = overlay[top:bottom, left:right]
        zoomed_original = original_np[top:bottom, left:right]

        tab_zoom, tab_full = st.tabs(["🔎 Preview zoom (area dianalisis)", "🖼️ Foto penuh"])
        with tab_zoom:
            zcol1, zcol2 = st.columns(2)
            with zcol1:
                st.image(zoomed_original, caption="Area asli (di-zoom)", use_container_width=True)
            with zcol2:
                st.image(
                    zoomed_overlay,
                    caption="Area dipilih (hijau) + flash dibuang",
                    use_container_width=True,
                )
        with tab_full:
            st.image(
                overlay,
                caption=f"{result['n_pixels_total']} piksel dipilih, "
                        f"{result['n_pixels_flash']} dibuang sebagai flash",
                use_container_width=True,
            )

        n_valid = result["n_pixels_total"] - result["n_pixels_flash"]
        st.caption(
            f"Piksel valid dipakai untuk kalkulasi: **{n_valid}** "
            f"(dari {result['n_pixels_total']} dicat, "
            f"{result['n_pixels_flash']} dibuang karena flash)"
        )

        label = result["prediction_label"]
        proba = result["prediction_proba"]

        if label.lower() == "yes":
            confidence = proba.get("Yes", 0)
            label_display = "Anemic"
        else:
            confidence = proba.get("No", 0)
            label_display = "Non-anemic"

        gcol, bcol = st.columns(2)
        with gcol:
            st.plotly_chart(make_gauge(confidence, label_display), use_container_width=True)
        with bcol:
            st.plotly_chart(make_rgb_bar(result["pct_red"], result["pct_green"], result["pct_blue"]), use_container_width=True)

        if label.lower() == "yes":
            st.error(f"⚠️ Hasil: **Anemic** (confidence: {confidence:.1%})")
        else:
            st.success(f"✅ Hasil: **Non-anemic** (confidence: {confidence:.1%})")

        with st.expander("Detail confidence score (angka)"):
            st.json(proba)

        entry = {
            "name": user_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": uploaded_file.name,
            "sex": sex,
            "pct_red": result["pct_red"],
            "pct_green": result["pct_green"],
            "pct_blue": result["pct_blue"],
            "n_pixels_total": result["n_pixels_total"],
            "n_pixels_flash": result["n_pixels_flash"],
            "prediction_label_display": label_display,
            "confidence": confidence,
            "prediction_proba": proba,
            "warnings": warnings,
        }
        add_to_history(entry)

        st.download_button(
            "⬇️ Download laporan hasil ini (.txt)",
            data=build_report_text(entry),
            file_name=f"laporan_anemia_{user_name.replace(' ', '_')}_{entry['timestamp'].replace(':', '-').replace(' ', '_')}.txt",
            mime="text/plain",
        )

    except ValueError as e:
        st.warning(f"Gagal memproses: {e}")

# ----------------------------------------------------------------------
# HISTORY DALAM SESI INI
# ----------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    st.subheader("📜 Riwayat sesi ini")
    st.caption(
        "Riwayat ini hanya tersimpan selama browser/tab tidak ditutup "
        "(tidak disimpan permanen ke database)."
    )

    if st.button("🗑️ Hapus semua riwayat"):
        st.session_state.history = []
        st.rerun()

    for i, entry in enumerate(st.session_state.history):
        icon = "⚠️" if entry["prediction_label_display"] == "Anemic" else "✅"
        with st.expander(
            f"{icon} {entry['timestamp']} — {entry['name']} — {entry['filename']} — "
            f"{entry['prediction_label_display']} ({entry['confidence']:.1%})"
        ):
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("%Red", f"{entry['pct_red']:.2f}%")
            hc2.metric("%Green", f"{entry['pct_green']:.2f}%")
            hc3.metric("%Blue", f"{entry['pct_blue']:.2f}%")
            st.write(f"Jenis kelamin: **{entry['sex']}**")
            if entry["warnings"]:
                for w in entry["warnings"]:
                    st.caption(f"⚠️ {w}")
            st.download_button(
                "⬇️ Download laporan ini",
                data=build_report_text(entry),
                file_name=f"laporan_anemia_{i}.txt",
                mime="text/plain",
                key=f"dl_hist_{i}",
            )

st.divider()
st.caption(
    "💡 Tips: cat hanya bagian konjungtiva yang merah (hindari sklera "
    "putih, bulu mata, dan kulit) untuk hasil yang paling akurat. "
    "Gunakan tombol bin/undo di toolbar kanvas kalau salah mencat."
)
