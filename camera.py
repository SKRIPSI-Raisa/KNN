import os
import cv2
import joblib
import numpy as np
import time
from skimage.feature import graycomatrix, graycoprops

# Konfigurasi
IMAGE_SIZE = (128, 128)
BOX_SIZE = 250  # Ukuran kotak target ROI

# Colors (BGR)
COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 50, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (150, 150, 150)
COLOR_BLUE = (255, 120, 0)

def extract_rgb_features(image_rgb):
    """
    Mengekstrak rata-rata dan standar deviasi dari channel R, G, B.
    Sesuai dengan implementasi di notebook pelatihan.
    """
    features = []
    for c in range(3):
        channel = image_rgb[:, :, c]
        features.append(np.mean(channel))
        features.append(np.std(channel))
    return features

def extract_glcm_features(gray_image):
    """
    Mengekstrak fitur tekstur GLCM (Contrast, Energy, Homogeneity, Correlation).
    Sesuai dengan implementasi di notebook pelatihan.
    """
    glcm = graycomatrix(
        gray_image,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    contrast = graycoprops(glcm, "contrast").mean()
    energy = graycoprops(glcm, "energy").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    correlation = graycoprops(glcm, "correlation").mean()
    return [contrast, energy, homogeneity, correlation]

def main():
    # 1. Load Model, Scaler, dan Label Encoder
    print("="*60)
    print("   PENGUJIAN REAL-TIME DENGAN KAMERA (KNN + GLCM + RGB)")
    print("="*60)
    
    # Gunakan path absolut berdasarkan lokasi file camera.py agar program bisa dijalankan dari folder mana saja
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "knn_best.pkl")
    scaler_path = os.path.join(script_dir, "scaler.pkl")
    le_path = os.path.join(script_dir, "label_encoder.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)):
        print("[ERROR] File model (.pkl) tidak lengkap!")
        print("Pastikan file berikut ada di folder yang sama dengan camera.py:")
        print(f" - {model_path}")
        print(f" - {scaler_path}")
        print(f" - {le_path}")
        return

    print("Memuat file model...")
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        print("Model, Scaler, dan Label Encoder berhasil dimuat!")
        print(f"Kelas terdaftar: {list(le.classes_)}")
    except Exception as e:
        print(f"[ERROR] Gagal memuat file model: {e}")
        return

    # 2. Inisialisasi Kamera
    print("\nMembuka kamera...")
    cap = None
    
    # Mencoba beberapa indeks kamera dan backend (prioritaskan DSHOW di Windows)
    camera_indices = [0, 1, 2]
    backends = [cv2.CAP_DSHOW, None] if os.name == 'nt' else [None]
    
    for idx in camera_indices:
        for backend in backends:
            try:
                if backend is not None:
                    print(f"Mencoba membuka kamera indeks {idx} dengan backend DSHOW...")
                    temp_cap = cv2.VideoCapture(idx, backend)
                else:
                    print(f"Mencoba membuka kamera indeks {idx} dengan backend default...")
                    temp_cap = cv2.VideoCapture(idx)
                
                if temp_cap.isOpened():
                    # Lakukan test read frame untuk memastikan kamera benar-benar menghasilkan gambar
                    ret, test_frame = temp_cap.read()
                    if ret and test_frame is not None:
                        cap = temp_cap
                        print(f"Kamera berhasil dibuka pada indeks {idx}!")
                        break
                    else:
                        temp_cap.release()
            except Exception:
                pass
        if cap is not None:
            break
            
    if cap is None:
        print("[ERROR] Gagal membuka kamera.")
        print("Solusi:")
        print(" 1. Pastikan kamera terhubung (webcam eksternal/internal).")
        print(" 2. Pastikan kamera tidak sedang digunakan oleh aplikasi lain (Zoom, Teams, Discord, dll).")
        print(" 3. Pastikan izin akses kamera di Settings Windows (Privacy & Security -> Camera) sudah aktif.")
        return

    print("KONTROL PADA WINDOW KAMERA:")
    print(" -> Tekan tombol 'T' : Mengaktifkan/menonaktifkan mode kotak target (ROI)")
    print(" -> Tekan tombol 'Q' : Keluar dari program")
    print("="*60)

    use_roi = True
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal mengambil frame dari kamera.")
            break

        # Mirror frame untuk visualisasi natural (webcam mode)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Tentukan koordinat kotak ROI (Region of Interest) di tengah layar
        x1 = int((w - BOX_SIZE) / 2)
        y1 = int((h - BOX_SIZE) / 2)
        x2 = x1 + BOX_SIZE
        y2 = y1 + BOX_SIZE

        # Ambil region gambar yang akan diproses
        if use_roi:
            process_area = frame[y1:y2, x1:x2]
        else:
            process_area = frame

        # Ekstraksi fitur & Prediksi
        try:
            # Resize ke ukuran training
            img_resized = cv2.resize(process_area, IMAGE_SIZE)
            rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            # Hitung fitur RGB dan GLCM
            rgb_feats = extract_rgb_features(rgb)
            glcm_feats = extract_glcm_features(gray)
            features = rgb_feats + glcm_feats

            # Scaling fitur
            features_scaled = scaler.transform([features])
            
            # Prediksi kelas
            pred_class_idx = model.predict(features_scaled)[0]
            pred_class = le.inverse_transform([pred_class_idx])[0]
            
            # Hitung probabilitas/confidence (jika model KNN mendukung)
            try:
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = probabilities[pred_class_idx] * 100
            except Exception:
                confidence = 100.0  # Fallback
        except Exception as e:
            pred_class = "Error"
            confidence = 0.0
            rgb_feats = [0]*6
            glcm_feats = [0]*4

        # Hitung FPS real-time
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time

        # --- DRAW VISUAL INTERFACE (Premium GUI Overlay) ---
        # 1. Overlay Panel Kiri (Semi-transparan hitam untuk readability)
        panel_width = 320
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # 2. Garis pembatas panel
        cv2.line(frame, (panel_width, 0), (panel_width, h), COLOR_GRAY, 1)

        # 3. Konten Panel Kiri
        # Judul Program
        cv2.putText(frame, "DETEKTOR SAMPAH (KNN)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLUE, 2)
        cv2.line(frame, (20, 48), (panel_width - 20, 48), COLOR_GRAY, 1)

        # Mode & Status FPS
        mode_str = "ROI (Kotak Tengah)" if use_roi else "Full Screen"
        cv2.putText(frame, f"Mode: {mode_str}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        cv2.line(frame, (20, 110), (panel_width - 20, 110), COLOR_GRAY, 1)

        # Bagian Prediksi
        cv2.putText(frame, "PREDIKSI:", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
        
        # Warna teks sesuai label prediksi
        class_color = COLOR_GREEN if pred_class.lower() == "organik" else COLOR_RED
        if pred_class == "Error":
            class_color = COLOR_WHITE
            
        cv2.putText(frame, pred_class.upper(), (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, class_color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 1)
        cv2.line(frame, (20, 205), (panel_width - 20, 205), COLOR_GRAY, 1)

        # Nilai Fitur Real-time
        cv2.putText(frame, "FITUR EKSTRAKSI:", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
        
        y_offset = 260
        # Fitur Warna (RGB)
        cv2.putText(frame, f"R mean/std: {rgb_feats[0]:.1f} / {rgb_feats[1]:.1f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
        cv2.putText(frame, f"G mean/std: {rgb_feats[2]:.1f} / {rgb_feats[3]:.1f}", (20, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
        cv2.putText(frame, f"B mean/std: {rgb_feats[4]:.1f} / {rgb_feats[5]:.1f}", (20, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
        
        # Fitur Tekstur (GLCM)
        cv2.putText(frame, f"Contrast   : {glcm_feats[0]:.4f}", (20, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
        cv2.putText(frame, f"Energy     : {glcm_feats[1]:.4f}", (20, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
        cv2.putText(frame, f"Homogeneity: {glcm_feats[2]:.4f}", (20, y_offset + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
        cv2.putText(frame, f"Correlation: {glcm_feats[3]:.4f}", (20, y_offset + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
        
        cv2.line(frame, (20, y_offset + 150), (panel_width - 20, y_offset + 150), COLOR_GRAY, 1)

        # Instruksi Pintasan
        cv2.putText(frame, "Pintasan Keyboard:", (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)
        cv2.putText(frame, "[T] Toggle Mode (ROI/Full)", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
        cv2.putText(frame, "[Q] Keluar Aplikasi", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

        # 4. Gambar Kotak Target ROI di Center (jika mode ROI aktif)
        if use_roi:
            # Gambar bracket sudut yang stylish daripada kotak utuh
            bracket_len = 20
            thickness = 2
            
            # Top-Left Corner
            cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), class_color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), class_color, thickness)
            # Top-Right Corner
            cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), class_color, thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), class_color, thickness)
            # Bottom-Left Corner
            cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), class_color, thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), class_color, thickness)
            # Bottom-Right Corner
            cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), class_color, thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), class_color, thickness)

            # Label panduan di bawah kotak
            cv2.putText(frame, "Posisikan sampah di dalam kotak", (x1 - 10, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

        # Tampilkan frame
        cv2.imshow("Detektor Sampah - Kamera Real-Time", frame)

        # Cek input keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('t') or key == ord('T'):
            use_roi = not use_roi

    # Bersihkan resource
    cap.release()
    cv2.destroyAllWindows()
    print("Kamera ditutup. Selesai.")

if __name__ == "__main__":
    main()
