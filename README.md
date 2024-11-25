# **ردیابی دست و کنترل نشانگر ماوس با استفاده از Mediapipe و OpenCV**

این پروژه یک سیستم ردیابی دست مبتنی بر پایتون را نشان می‌دهد که از ماژول **MediaPipe Hands** برای کنترل نشانگر ماوس با دنبال کردن حرکت انگشت اشاره در زمان واقعی استفاده می‌کند.

## **ویژگی‌ها**
- ردیابی حرکات دست با استفاده از وبکم.
- حرکت نشانگر ماوس بر اساس موقعیت انگشت اشاره.
- نمایش لحظه‌ای نقاط کلیدی دست.

## **پیش‌نیازها**
برای اجرای این پروژه به کتابخانه‌های زیر نیاز دارید:
- `mediapipe`
- `opencv-python`
- `pyautogui`

همچنین باید یک وبکم فعال به سیستم خود متصل کرده باشید.

## **نصب**
1. مخزن را کلون کنید:
   ```bash
   git clone https://github.com/<your-username>/<your-repository>.git
   cd <your-repository>
   ```
2. کتابخانه‌های موردنیاز را نصب کنید:
   ```bash
   pip install mediapipe opencv-python pyautogui
   ```

## **نحوه اجرا**
1. اسکریپت را اجرا کنید:
   ```bash
   python MouseControl.py
   ```
2. برنامه شروع به ضبط ویدئو از وبکم می‌کند. دست خود را جلوی دوربین قرار دهید تا شناسایی شود.
3. با حرکت انگشت اشاره، نشانگر ماوس را روی صفحه کنترل کنید.

## **نحوه عملکرد**
- برنامه از ماژول Hands در Mediapipe برای شناسایی نقاط کلیدی دست استفاده می‌کند.
- مختصات نوک انگشت اشاره استخراج شده و با استفاده از `pyautogui` به ابعاد صفحه نمایش نگاشت می‌شود.
- OpenCV برای ضبط ویدئو در زمان واقعی و نمایش نقاط کلیدی استفاده می‌شود.

## **کلیدهای کنترلی**
- **کلید ESC**: خروج از برنامه.

---

در صورت تمایل به مشارکت در این پروژه یا گزارش مشکلات، خوشحال می‌شویم که کمک کنید! 😊