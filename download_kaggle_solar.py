import os
import subprocess
import sys

def install_and_download():
    print("กำลังติดตั้ง Library ที่จำเป็น...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle", "pandas"])
    
    # ตรวจสอบหาไฟล์ API Token ของ Kaggle
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("\n❌ ไม่พบไฟล์ kaggle.json สำหรับยืนยันตัวตน!")
        print("Kaggle มีระบบป้องกันที่เข้มงวดมาก ต้องใช้ API Key ในการโหลดครับ")
        print("="*60)
        print("👉 วิธีเอาไฟล์ kaggle.json (ฟรีและเร็วมาก):")
        print("1. เข้าเว็บ https://www.kaggle.com/ ล็อคอินให้เรียบร้อย")
        print("2. กดที่รูปโปรไฟล์ขวาบน -> เลือก 'Settings'")
        print("3. เลื่อนลงมาที่หมวด 'API' แล้วกดปุ่ม 'Create New Token'")
        print("4. ไฟล์ kaggle.json จะถูกโหลดลงเครื่องของคุณ")
        print(f"5. ให้คุณสร้างโฟลเดอร์ {kaggle_dir} และนำไฟล์ไปวางไว้ข้างใน")
        print("   (ใน Mac/Linux สามารถพิมพ์: mkdir ~/.kaggle แล้วลากไฟล์ไปใส่)")
        print("6. รันคำสั่ง chmod 600 ~/.kaggle/kaggle.json เพื่อป้องกันสิทธิ์")
        print("7. กลับมารันคำสั่ง python download_kaggle_solar.py อีกครั้ง!")
        print("="*60)
        return
        
    print("\n✅ พบไฟล์ kaggle.json เริ่มทำการดาวน์โหลด...")
    
    save_dir = "data/1_raw/kaggle_solar"
    os.makedirs(save_dir, exist_ok=True)
    
    dataset_name = "anikannal/solar-power-generation-data"
    
    print(f"กำลังดึงข้อมูลชุด: {dataset_name} ...")
    cmd = ["kaggle", "datasets", "download", "-d", dataset_name, "-p", save_dir, "--unzip"]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n🎉 ดาวน์โหลดและแตกไฟล์สำเร็จ! ข้อมูลอยู่ที่: {save_dir}")
        print("\nไฟล์ที่คุณได้คือ:")
        for f in os.listdir(save_dir):
            if f.endswith(".csv"):
                print(f" - {f}")
                
        print("\n🚀 เมื่อโหลดสำเร็จ คุณจะต้องนำไฟล์ Plant_1_Generation_Data.csv")
        print("และ Plant_1_Weather_Sensor_Data.csv มา Merge (Join) กันตามคอลัมน์ DATE_TIME ก่อนนำไปรันครับ")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ เกิดข้อผิดพลาดในการดาวน์โหลดจาก Kaggle: {e}")

if __name__ == "__main__":
    install_and_download()
