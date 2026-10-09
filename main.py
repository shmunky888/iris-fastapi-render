
# main.api.py
# นี่คือไฟล์ที่จะใช้รันเป็น Web Server ของเรา

from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

# สร้างแอป FastAPI พร้อมใส่ Metadata ที่สวยงาม
app = FastAPI(
    title="Iris Species Prediction API",
    description="An API to predict the species of Iris flowers. Created for educational purposes.",
    version="1.0.0"
)

# โหลดโมเดลที่ฝึกไว้
# โค้ดนี้จะทำงานเมื่อ Server เริ่มต้นขึ้น
try:
    model = load('iris_random_forest.joblib')
    target_names = ['setosa', 'versicolor', 'virginica']
except FileNotFoundError:
    model = None
    target_names = []

# กำหนดโครงสร้างข้อมูล Input ที่จะรับเข้ามาผ่าน API
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# สร้าง Endpoint พื้นฐานสำหรับทดสอบ
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Prediction API! Go to /docs to see the documentation."}

# สร้าง Endpoint สำหรับการทำนาย (/predict)
# @app.post หมายถึงรับข้อมูลผ่าน HTTP POST method
@app.post("/predict")
def predict_iris(data: IrisData):
    if model is None:
        return {"error": "Model not found."}

    # แปลงข้อมูลจาก API เป็น numpy array ที่โมเดลเข้าใจ
    input_data = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    # ทำนายผล
    prediction_index = model.predict(input_data)[0]
    predicted_class_name = target_names[prediction_index]

    # ส่งผลลัพธ์กลับไปในรูปแบบ JSON
    return {
        "input": data.dict(),
        "predicted_class_index": int(prediction_index),
        "predicted_class_name": predicted_class_name
    }
