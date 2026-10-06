from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import telebot
import tensorflow as tf

app = FastAPI()

# --- KONFIGURASI TELEGRAM BOT (Sesuai data Anda) ---
TOKEN = "8585090685:AAGDOeoROagyuZYOGXuD35WGg4rARKU1tXw"
CHAT_ID = "7212316891"
bot = telebot.TeleBot(TOKEN)

# --- LOAD MODEL TENSORFLOW ---
# Mengarah ke folder hasil ekstrak file ZIP Anda
MODEL_PATH = "ei-semoga-berhasil-transfer-learning-tensorflow-savedmodel-model.7" 

print("Memuat model TensorFlow...")
try:
    model = tf.saved_model.load(MODEL_PATH)
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

        # 3. Proses Prediksi Model SavedModel TensorFlow
        infer = model.signatures["serving_default"]
        predictions = infer(tf.constant(input_data))
        
        result_key = list(predictions.keys())[0]
        scores = predictions[result_key].numpy()[0]
        
        predicted_class_index = np.argmax(scores)
        confidence = float(scores[predicted_class_index])

        # Daftar kelas (Masker, Uang 50rb, Obat)
        classes = ["masker", "uang_50rb", "obat"]
        predicted_label = classes[predicted_class_index]

        # 4. Kirim hasil notifikasi ke Telegram Anda
        pesan = f"🚨 *Deteksi Objek ESP32-CAM*\n\n- Hasil: *{predicted_label}*\n- Akurasi: {confidence * 100:.2f}%"
        bot.send_message(CHAT_ID, pesan, parse_mode="Markdown")

        return {
            "status": "success",
            "prediction": predicted_label,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}