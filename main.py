from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import telebot
import tensorflow as tf
import zipfile
import os

app = FastAPI()

# --- KONFIGURASI TELEGRAM ---
TOKEN = "8585090685:AAGDOeoROagyuZYOGXuD35WGg4rARKU1tXw"
CHAT_ID = "7212316891"
bot = telebot.TeleBot(TOKEN)

# --- OTOMATIS EKSTRAK ZIP DI SERVER CLOUD ---
ZIP_FILENAME = "ei-semoga-berhasil-transfer-learning-tensorflow-savedmodel-model.7" # Sesuaikan dengan nama file zip Anda di GitHub
EXTRACT_PATH = "model_extracted"

print("Mengekstrak file model...")
try:
    if os.path.exists(ZIP_FILENAME):
        with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        print("File zip berhasil diekstrak!")
    else:
        print(f"File {ZIP_FILENAME} tidak ditemukan!")
except Exception as e:
    print(f"Gagal ekstrak: {e}")

# --- LOAD MODEL TENSORFLOW ---
print("Memuat model TensorFlow...")
try:
    # Mengarah ke folder hasil ekstrak otomatis
    model = tf.saved_model.load(f"{EXTRACT_PATH}/saved_model")
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model: {e}")

@app.get("/")
def read_root():
    return {"status": "Server AI ESP32-CAM Aktif!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Membaca gambar dari ESP32-CAM
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Gagal mendecode gambar"}

        # 2. Preprocessing gambar (Ukuran 96x96 standar Edge Impulse)
        img_resized = cv2.resize(img, (96, 96)) 
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32)

        # 3. Proses Prediksi Model
        infer = model.signatures["serving_default"]
        predictions = infer(tf.constant(input_data))
        
        result_key = list(predictions.keys())[0]
        scores = predictions[result_key].numpy()[0]
        
        predicted_class_index = np.argmax(scores)
        confidence = float(scores[predicted_class_index])

        # Daftar kelas (Masker, Uang 50rb, Obat)
        classes = ["masker", "uang_50rb", "obat"]
        predicted_label = classes[predicted_class_index] if predicted_class_index < len(classes) else "Unknown"

        # 4. Kirim notifikasi ke Telegram
        pesan = f"🚨 *Deteksi Objek ESP32-CAM*\n\n- Hasil: *{predicted_label}*\n- Akurasi: {confidence * 100:.2f}%"
        bot.send_message(CHAT_ID, pesan, parse_mode="Markdown")

        return {
            "status": "success",
            "prediction": predicted_label,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}
