import os
import pandas as pd
import urllib.request

def download_solar_data():
    # URL สำหรับ Open Power System Data (ประเทศเยอรมนี) 
    # ข้อมูลประกอบด้วยปริมาณการใช้ไฟฟ้า (Consumption), พลังงานลม (Wind) และ พลังงานแสงอาทิตย์ (Solar)
    url = "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
    
    save_dir = "data/1_raw/solar_dataset"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "opsd_germany_daily.csv")
    
    print(f"กำลังดาวน์โหลดข้อมูล Solar Data จาก OPSD เยอรมนี...")
    print(f"URL: {url}")
    
    try:
        # ดาวน์โหลดไฟล์ CSV
        urllib.request.urlretrieve(url, save_path)
        print(f"\n✅ ดาวน์โหลดสำเร็จ! บันทึกไฟล์ไว้ที่: {save_path}")
        
        # โหลดมาพรีวิวดู
        df = pd.read_csv(save_path)
        print(f"จำนวนข้อมูลทั้งหมด: {df.shape[0]} วัน (แถว), {df.shape[1]} คอลัมน์")
        print("รายชื่อคอลัมน์:", df.columns.tolist())
        
        print("\nตัวอย่างข้อมูล 3 บรรทัดแรก:")
        print(df.head(3).to_markdown())
        
        print("\n" + "="*60)
        print("🚀 สิ่งที่คุณต้องทำต่อไป:")
        print("เปิดไฟล์ config.json แล้วแก้ค่าให้เป็นตามนี้ครับ:")
        print('  "raw_data_path": "data/1_raw/solar_dataset/opsd_germany_daily.csv"')
        print('  "time_column": "Date"')
        print('  "resample_freq": "1D"  <-- (เปลี่ยนเป็น 1D เพราะข้อมูลชุดนี้เก็บรายวัน)')
        print('  "target_source_column": "Solar"')
        print('  "exclude_features": ["Solar"]')
        print("="*60)
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดาวน์โหลด: {e}")

if __name__ == "__main__":
    download_solar_data()
