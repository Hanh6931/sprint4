from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os
import time
import nbformat
from nbconvert import PythonExporter
from flask_cors import CORS, cross_origin
import logging
from flask import Flask, jsonify
import subprocess
import platform
import mysql.connector
from flask_sqlalchemy import SQLAlchemy
logging.basicConfig(level=logging.DEBUG)


# 创建 Flask 应用，并指定模板文件夹为 'html_templates'
app = Flask(__name__, template_folder='html_templates', static_folder='static')

# Configure Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://XUL3227:4C1LW181@wayne.cs.uwec.edu:3306/cs485group4'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Test Database Connection
def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host="wayne.cs.uwec.edu",
            user="XUL3227",
            password="4C1LW181",
            database="cs485group4",
            port=3306,
            connection_timeout=10
        )
        if connection.is_connected():
            print("Successfully connected to remote database: cs485group4")
        return connection
    except mysql.connector.Error as e:
        print(f"Error connecting to database: {e}")
        return None

if connect_to_database():
    print("Database connection successful!")
else:
    print("Database connection failed!")


# 加载 Wi-Fi 模型和标准化器
def load_wifi_model():
    try:
        svm_wifi_model = joblib.load('svm_wifi_localization_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("Models loaded successfully.")
    except FileNotFoundError:
        print("Model files not found. Training new models...")
        svm_wifi_model, scaler = preprocess_wifi_data()
    return svm_wifi_model, scaler

# 训练 Wi-Fi 模型（模拟，使用原有数据）
def preprocess_wifi_data():
    data = pd.read_csv('wifi_localization.txt', sep='\t', header=None)
    data.columns = [f"Signal_{i+1}" for i in range(7)] + ["Room"]
    X = data.drop("Room", axis=1)
    y = data["Room"]
    
    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练 SVM 模型
    svm_wifi_model = SVC(kernel='linear', random_state=42)
    svm_wifi_model.fit(X_train_scaled, y_train)
    
    # 保存模型
    joblib.dump(svm_wifi_model, 'svm_wifi_localization_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return svm_wifi_model, scaler

# 系统初始化
svm_wifi_model, scaler = load_wifi_model()
lights_brightness = {"Bedroom": 2, "Kitchen": 2, "Living Room": 2, "Bathroom": 2}  # 默认亮度
lights = {"Bedroom": 0, "Kitchen": 0, "Living Room": 0, "Bathroom": 0}  # 灯的状态

# 控制灯光
def control_light(state, room, lights, lights_brightness):
    if state == 'on':
        lights[room] = 1
        lights_brightness[room] = 2  # 默认亮度为2
        print(f"Turning on the light in {room}.")
    elif state == 'off':
        lights[room] = 0
        print(f"Turning off the light in {room}.")

# 控制空调
def control_ac(state, room):
    if state == 'on':
        print(f"Turning on the air conditioner in {room}.")
    elif state == 'off':
        print(f"Turning off the air conditioner in {room}.")

# 控制窗帘
def control_curtain(state, room):
    if state == 'open':
        print(f"Opening the curtain in {room}.")
    elif state == 'close':
        print(f"Closing the curtain in {room}.")

# 路由：主页
@app.route('/')
def index():
    return render_template('index.html')  # 渲染模板文件

@app.route('/signup.html')
def signup():
    return render_template('signup.html') 

@app.route('/forgot-password.html')
def forgetpassword():
    return render_template('forgot-password.html') 
    
@app.route('/floorplan.html')
def floorplan():
    return render_template('floorplan.html')

# 渲染客厅页面
@app.route('/living-room.html')
def livingroom():
    return render_template('living-room.html')

# 渲染厨房页面
@app.route('/kitchen.html')
def kitchen():
    return render_template('kitchen.html')

# 渲染卧室页面
@app.route('/bedroom.html')
def bedroom():
    return render_template('bedroom.html')

# 渲染卫生间页面
@app.route('/bathroom.html')
def bathroom():
    return render_template('bathroom.html')

# 路由：控制灯光开关
@app.route('/control_light', methods=['POST'])
def control_light_route():
    room = request.form.get('room', '').lower()
    state = request.form.get('state', '').lower()
    control_light(state, room, lights, lights_brightness)
    return jsonify({
        "status": "success",
        "room": room,
        "state": state,
        "visual_effect": "on" if lights[room] == 1 else "off"
    })


# 路由：控制空调
@app.route('/control_curtain', methods=['POST'])
def control_curtain_route():
    room = request.form.get('room', '').lower()
    state = request.form.get('state', '').lower()
    control_curtain(state, room)
    return jsonify({
        "status": "success",
        "room": room,
        "state": state,
        "visual_effect": "open" if state == "open" else "close"
    })

# 路由：控制空调
@app.route('/control_ac', methods=['POST'])
def control_ac_route():
    room = request.form.get('room', '').lower()
    state = request.form.get('state', '').lower()
    control_ac(state, room)
    return jsonify({
        "status": "success",
        "room": room,
        "state": state,
        "message": f"Air conditioner in {room} turned {state}."
    })




# 控制设备的API接口
@app.route('/control-device', methods=['POST'])
def control_device():
    data = request.get_json()
    room = data['room']
    device = data['device']
    action = data['action']
    
    # 更新设备状态
    devices[room][device] = action

    # 输出到控制台
    print(f"Room: {room}, Device: {device}, Action: {action}")
    
    return jsonify({'status': 'success'})

# 语音控制接口
@app.route('/voice-control', methods=['POST'])
def voice_control():
    data = request.get_json()  # 获取从前端发送的JSON数据
    command = data.get('command', '').lower()  # 提取语音命令

    # 处理不同房间的灯光、空调和窗帘控制命令
    # （此处省略语音控制命令处理代码，保持现有逻辑）
    
    return jsonify({'status': 'success', 'message': f'Executed command: {command}'})

# 路由：运行Notebook文
@app.route('/run_script', methods=['POST'])
def run_script():
    try:
        # 根据操作系统选择命令行程序
        if platform.system() == "Windows":
            # Windows: 使用 'start' 命令打开新的命令行窗口
            subprocess.Popen(["start", "cmd", "/k", "python temp_script.py"], shell=True)
        elif platform.system() == "Linux" or platform.system() == "Darwin":
            # Linux 和 macOS: 使用 'xterm' 打开终端窗口
            subprocess.Popen(["xterm", "-e", "python3 temp_script.py"])

        # 返回响应表示命令成功启动
        return jsonify({"message": "Script is running in a new terminal window"}), 200
    except Exception as e:
        # 捕获异常并返回错误信息
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
