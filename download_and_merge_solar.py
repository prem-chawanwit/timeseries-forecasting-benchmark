import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os

def prepare_solar_data():
    print("ดาวน์โหลด Solar Power Generation Data ผ่าน kagglehub...")
    
    # 1. โหลดข้อมูลการผลิตไฟ (Generation Data)
    print("-> Loading Generation Data...")
    df_gen = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "anikannal/solar-power-generation-data",
        "Plant_1_Generation_Data.csv"
    )
    print(f"Generation Data Shape: {df_gen.shape}")

    # 2. โหลดข้อมูลสภาพอากาศ (Weather Sensor Data)
    print("-> Loading Weather Data...")
    df_weather = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "anikannal/solar-power-generation-data",
        "Plant_1_Weather_Sensor_Data.csv"
    )
    print(f"Weather Data Shape: {df_weather.shape}")

    print("\nกำลังทำการ Merge (ผูกข้อมูลเข้าด้วยกันด้วยเวลา DATE_TIME)...")
    
    # แปลงคอลัมน์ DATE_TIME เป็น datetime เพื่อให้ตรงกัน
    df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'], format="%d-%m-%Y %H:%M")
    df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'], format="%Y-%m-%d %H:%M:%S")
    
    # รวมตาราง (Inner Join)
    df_merged = pd.merge(df_gen, df_weather, on=['DATE_TIME', 'PLANT_ID'], how='inner')
    print(f"Merged Data Shape: {df_merged.shape}")
    
    print("\nกำลังทำการรวมพลังงานแผงโซลาร์ทั้งหมดในโรงไฟฟ้าเข้าด้วยกัน...")
    # ในโรงไฟฟ้า 1 โรง มี Inverter หลายตัว (SOURCE_KEY)
    # เราจะทำการ Sum ไฟที่ผลิตได้รวมทั้งหมด และหาค่าเฉลี่ยของสภาพอากาศในเวลานั้นๆ
    df_agg = df_merged.groupby('DATE_TIME').agg({
        'DC_POWER': 'sum',
        'AC_POWER': 'sum',
        'DAILY_YIELD': 'sum',
        'TOTAL_YIELD': 'sum',
        'AMBIENT_TEMPERATURE': 'mean',
        'MODULE_TEMPERATURE': 'mean',
        'IRRADIATION': 'mean'
    }).reset_index()
    
    # เรียงลำดับเวลาให้ถูกต้อง
    df_agg = df_agg.sort_values(by='DATE_TIME').reset_index(drop=True)
    print(f"Aggregated Time Series Shape: {df_agg.shape}")
    
    # สร้างโฟลเดอร์สำหรับเก็บข้อมูล
    save_dir = "data/1_raw"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "kaggle_solar_merged.csv")
    
    # บันทึกไฟล์
    df_agg.to_csv(save_path, index=False)
    print(f"\n✅ บันทึกไฟล์ข้อมูลพร้อมใช้งานเรียบร้อยแล้วที่: {save_path}")
    print("\nคอลัมน์ที่มีให้ใช้เป็น Feature (เยอะจุใจแน่นอน):")
    for col in df_agg.columns:
        print(f" - {col}")

if __name__ == "__main__":
    prepare_solar_data()
