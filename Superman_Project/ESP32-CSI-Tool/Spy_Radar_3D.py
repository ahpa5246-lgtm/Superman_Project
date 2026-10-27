import threading
import time
import math
import numpy as np
from ursina import *

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ==========================================
# 1. قارئ الموجات (الرادار الخلفي)
# ==========================================
class RadarListener(threading.Thread):
    def __init__(self, port="COM3", baud=115200):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.activity_level = 0.0  # مستوى الحركة في الغرفة (0.0 إلى 1.0)
        self.running = True

    def run(self):
        if SERIAL_AVAILABLE:
            try:
                ser = serial.Serial(self.port, self.baud, timeout=1)
                while self.running:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if "CSI_DATA:" in line:
                        payload = line.split(":")[1]
                        raw = np.fromstring(payload, dtype=np.int32, sep=",")
                        if raw.size > 10:
                            amps = np.sqrt(raw[0::2]**2 + raw[1::2]**2)
                            # حساب التباين (Variance) لمعرفة مدى اضطراب الموجات بسبب الحركة
                            variance = np.var(amps)
                            # تحويل الاضطراب إلى نسبة مئوية (نشاط)
                            target_activity = min(variance / 200.0, 1.0)
                            # جعل الانتقال ناعماً
                            self.activity_level = (self.activity_level * 0.8) + (target_activity * 0.2)
                return
            except:
                pass # في حال فشل الاتصال، سننتقل للمحاكاة فوراً

        # وضع المحاكاة (Demo) إذا لم تكن اللوحة متصلة
        t = 0
        while self.running:
            time.sleep(0.1)
            t += 0.1
            # توليد حركة وهمية تشبه مرور شخص كل بضع ثوانٍ
            simulated_movement = math.sin(t) * math.cos(t * 0.5)
            if simulated_movement > 0.7:
                self.activity_level = 1.0 # شخص يتحرك
            else:
                self.activity_level = max(0.0, self.activity_level - 0.1) # سكون

# ==========================================
# 2. بناء الغرفة ثلاثية الأبعاد (غرفة العمليات)
# ==========================================
app = Ursina()
window.title = "SUPERMAN EYES - 3D RADAR"
window.color = color.black
window.fullscreen = False

# شبكة الأرضية (Matrix Grid)
grid = Entity(model='wireframe_cube', scale=(15, 0.1, 15), color=color.rgba(0, 255, 0, 100), y=-2)

# بناء مجسم بشري مبسط (Avatar)
avatar_pivot = Entity(y=0)
body = Entity(parent=avatar_pivot, model='cube', scale=(1, 1.5, 0.5), color=color.dark_gray, y=0)
head = Entity(parent=avatar_pivot, model='sphere', scale=(0.6, 0.6, 0.6), color=color.dark_gray, y=1.2)

# هالة التوهج المحيطة بالمجسم (تعكس قوة إشارة الواي فاي)
aura = Entity(parent=avatar_pivot, model='cube', scale=(1.2, 2.8, 1), color=color.rgba(0, 255, 0, 50))

# نصوص الواجهة الجاسوسية (HUD)
ui_title = Text(text="[ 3D SPY RADAR ACTIVE ]", origin=(0, 0), y=0.45, scale=1.5, color=color.lime)
ui_status = Text(text="STATUS: SCANNING SECTOR...", origin=(0, 0), y=0.40, scale=1.2, color=color.green)
ui_data = Text(text="VARIANCE: 0.00%", origin=(-0.5, 0), x=-0.8, y=0.45, scale=1, color=color.lime)

# الكاميرا السينمائية
camera.position = (0, 3, -10)
camera.rotation_x = 15

# تشغيل قارئ البيانات في الخلفية
radar = RadarListener(port="COM3")
radar.start()

# ==========================================
# 3. محرك التحديث الحي (يعمل 60 مرة في الثانية)
# ==========================================
def update():
    act = radar.activity_level
    
    # تحديث الأرقام على الشاشة
    ui_data.text = f"SIGNAL VARIANCE: {act * 100:.1f}%\nFREQ: 5GHz\nLINK: ACTIVE"

    # دوران الغرفة البطيء لإعطاء تأثير سينمائي
    grid.rotation_y += 5 * time.dt
    avatar_pivot.rotation_y -= 10 * time.dt

    # تفاعل المجسم مع حركة الواي فاي
    if act > 0.4:
        # شخص يتحرك! (توهج أحمر وحركة)
        body.color = color.red
        head.color = color.red
        aura.color = color.rgba(255, 0, 0, 150)
        aura.scale = (1.5 + act, 3.2 + act, 1.5 + act)
        
        ui_status.text = ">> ENTITY DETECTED IN ROOM <<"
        ui_status.color = color.red
        ui_title.color = color.red
    else:
        # الغرفة فارغة أو ساكنة
        body.color = color.dark_gray
        head.color = color.dark_gray
        aura.color = color.rgba(0, 255, 0, int(50 + act * 50))
        aura.scale = (1.2, 2.8, 1)
        
        ui_status.text = "STATUS: SCANNING SECTOR..."
        ui_status.color = color.green
        ui_title.color = color.lime

    # تأثير نبض الهالة (Breathing Effect)
    aura.scale += Vec3(math.sin(time.time() * 5) * 0.05, math.sin(time.time() * 5) * 0.05, math.sin(time.time() * 5) * 0.05)

# السماح بتحريك الكاميرا بالماوس (كليك يمين للتحريك)
EditorCamera()

app.run()