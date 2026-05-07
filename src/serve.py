from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# Cấu hình Cloud Provider và Bucket (Hỗ trợ cả AWS S3 và GCP GCS)
CLOUD_PROVIDER = os.environ.get("CLOUD_PROVIDER", "aws").lower()
CLOUD_BUCKET = os.environ.get("CLOUD_BUCKET") or os.environ.get("GCS_BUCKET") or os.environ.get("AWS_BUCKET")
MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """Tải file model.pkl từ Cloud Storage về máy khi server khởi động."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    if not CLOUD_BUCKET:
        raise ValueError("Chưa thiết lập tên Bucket qua biến môi trường CLOUD_BUCKET!")
        
    if CLOUD_PROVIDER == "gcp":
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(CLOUD_BUCKET)
        blob = bucket.blob(MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print("Model đã được tải xuống từ Google Cloud Storage.")
    else:  # Mặc định là AWS S3
        import boto3
        s3 = boto3.client("s3")
        s3.download_file(CLOUD_BUCKET, MODEL_KEY, MODEL_PATH)
        print("Model đã được tải xuống từ AWS S3.")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe server."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """Endpoint suy luận chính."""
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")
    pred = model.predict([req.features])[0]
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {
        "prediction": int(pred),
        "label": labels.get(int(pred), "unknown")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
