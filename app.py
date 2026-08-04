from fastapi import FastAPI, UploadFile
from predict import predict

app = FastAPI()

@app.post("/predict")
def predict_endpoint(file: UploadFile):
    image_bytes = file.file.read()
    tb_probability = predict(image_bytes)
    return {"tb_probability": tb_probability}