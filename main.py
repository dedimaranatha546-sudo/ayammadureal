import io
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

app = FastAPI(title="Klasifikasi Tanaman / Objek API")

# --- LOAD MODEL TENSORFLOW ---
print("Memuat model TensorFlow...")
try:
    # Memuat model langsung dari direktori utama tempat file main.py berada
    model = tf.saved_model.load(".")
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None


@app.get("/")
def home():
    return {
        "message": "API FastAPI Berjalan Normal!",
        "model_status": "Loaded" if model is not None else "Failed to load",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=500, detail="Model belum dimuat di server."
        )

    try:
        # Membaca file gambar yang diupload
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Sesuaikan ukuran input gambar jika diperlukan oleh model Anda (contoh: 224x224)
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Melakukan prediksi menggunakan SavedModel signature
        infer = model.signatures["serving_default"]
        input_tensor = tf.convert_to_tensor(img_array)
        predictions = infer(input_tensor)

        # Mengambil hasil output pertama
        output_key = list(predictions.keys())[0]
        result = predictions[output_key].numpy().tolist()

        return {"filename": file.filename, "prediction": result}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=fTerjadi kesalahan: {str(e)}"
        )
