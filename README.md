# Time Series Forecasting Benchmark

โครงสร้าง Folder นี้ออกแบบมาเพื่อใช้สำหรับการทดสอบโมเดล Time Series Forecasting 3 โมเดล ได้แก่ Baseline, XGBoost, และ LSTM โดยรองรับการทำ K-Fold Cross Validation และแยกสัดส่วนการทำงานชัดเจนเพื่อให้จัดการง่าย ไม่ซับซ้อนเกินไป

## Folder Structure

```text
.
├── data/                       # สำหรับจัดเก็บข้อมูลในแต่ละขั้นตอน (แยกชัดเจนตาม Pipeline)
│   ├── 1_raw/                  # เก็บข้อมูลดิบที่ดาวน์โหลดมา (เช่น จาก kaggle)
│   ├── 2_etl/                  # เก็บข้อมูลที่ผ่านขั้นตอน ETL (ทำความสะอาด, จัดการ Missing, สร้าง Features)
│   └── 3_kfold_splits/         # เก็บข้อมูลที่แบ่งเป็น Fold ย่อยๆ สำหรับ Train/Test ด้วยวิธี K-Fold CV
│
├── src/                        # โค้ดโปรแกรมทั้งหมด (แยกตามขั้นตอน)
│   ├── etl/                    # สคริปต์สำหรับดาวน์โหลด และทำ ETL
│   │   └── download_dataset.py # โค้ดสำหรับดาวน์โหลดข้อมูลผ่าน kagglehub
│   ├── models/                 # โครงสร้างและตั้งค่า (Hyperparameters) ของแต่ละโมเดล
│   ├── train/                  # โค้ดสำหรับรัน Training (รวมการรันลูป K-Fold CV ไว้ที่นี่)
│   │   ├── baseline/
│   │   ├── xgboost/
│   │   └── lstm/
│   └── test/                   # โค้ดสำหรับการโหลดโมเดลมารัน Testing/Evaluation ในขั้นตอนสุดท้าย
│
└── results/                    # สำหรับเก็บผลลัพธ์ของการรันโมเดลต่างๆ
    ├── baseline/               # เก็บไฟล์ Metrics (CSV/JSON), กราฟ (Plots), และ Weights (ถ้ามี)
    ├── xgboost/
    └── lstm/
```

## กระบวนการทำงานที่แนะนำ (Workflow)

1. **ETL Process:** 
   - รันสคริปต์ `python src/etl/process_and_split.py` เพื่อดึงและเตรียมข้อมูล 
   - แบ่งข้อมูลสำหรับการทำ K-Fold โดยจะได้ชุด `train.csv` และ `val.csv` ในแต่ละ Fold และชุด `test.csv` (Global Test) รวมศูนย์
2. **Experiment Tracking (รันทดสอบโมเดลทั้งหมด):**
   - รันสคริปต์ **`python run_experiment.py`**
   - ระบบจะสร้าง Folder ใหม่ให้โดยอัตโนมัติใน `experiments/run_YYYYMMDD_HHMMSS/`
   - โค้ดจะทยอยรัน Train ของ Baseline, XGBoost, LSTM จนครบ
   - โมเดลจะอ่านข้อมูล เลือก Features, ทำ Training และประเมินผล
3. **Evaluation & Result:**
   - เมื่อรันจบ ระบบจะสรุปผลและเซฟไฟล์ลงในโฟลเดอร์รันนั้นๆ เช่น:
      - `benchmark_comparison.png` (กราฟแท่ง)
      - `timeseries_comparison.png` (กราฟเส้นเปรียบเทียบผลพยากรณ์)
      - `statistical_tests.csv` (ผล Paired t-test ของแต่ละคู่โมเดล)

คุณสามารถนำโฟลเดอร์ของแต่ละการรัน (`experiments/run_...`) ไปวิเคราะห์ Lab ต่อได้อย่างง่ายดาย ไม่เกิดการเขียนทับผลลัพธ์เก่าครับ
