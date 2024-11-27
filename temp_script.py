#!/usr/bin/env python
# coding: utf-8

# In[2]:


import speech_recognition as sr
import joblib
import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import logging
logging.basicConfig(level=logging.DEBUG)


# 优化：检查 Wi-Fi 数据集路径是否存在
DATA_PATH = 'wifi_localization.txt'

def load_wifi_data(data_path):
    """加载Wi-Fi定位数据并进行预处理"""
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found.")
        return None

    print(f"Loading Wi-Fi localization data from {data_path}...")
    data = pd.read_csv(data_path, sep='\t', header=None)
    data.columns = [f"Signal_{i+1}" for i in range(7)] + ["Room"]
    return data

# 加载 Wi-Fi 室内定位模型和标准化器
def load_wifi_model():
    try:
        svm_wifi_model = joblib.load('svm_wifi_localization_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("Models loaded successfully.")
    except FileNotFoundError:
        print("Model files not found. Training new models...")
        svm_wifi_model, scaler = preprocess_wifi_data()  # 如果没有模型，则训练模型
    return svm_wifi_model, scaler


# 基于 Wi-Fi 信号预测房间
def predict_room(svm_wifi_model, scaler):
    # 模拟 Wi-Fi 信号
    wifi_signals = np.random.uniform(-90, -30, size=(1, 7))
    # 创建 DataFrame，并添加列名
    wifi_signals_df = pd.DataFrame(wifi_signals, columns=[f"Signal_{i+1}" for i in range(7)])
    # 标准化 Wi-Fi 信号
    wifi_signals_scaled = scaler.transform(wifi_signals_df)
    # 预测房间
    predicted_room = svm_wifi_model.predict(wifi_signals_scaled)[0]
    return predicted_room


# 获取实时环境数据（模拟）
def get_real_time_data():
    return {
        'Temperature': np.random.uniform(15, 30),
        'Humidity': np.random.uniform(30, 80),
        'Light_Intensity': np.random.uniform(0, 1),
        'Time_of_Day': np.random.randint(0, 24),
        'User_Home': np.random.randint(0, 2),
        'Weekday': np.random.randint(0, 2),
        'User_Preference': np.random.uniform(0, 1)
    }

# 训练并预处理 Wi-Fi 定位模型
def preprocess_wifi_data():
    data = load_wifi_data(DATA_PATH)
    if data is None:
        print("No data loaded. Exiting...")
        return None, None

    X = data.drop("Room", axis=1)
    y = data["Room"]
    
    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练SVM模型
    print("Training SVM model for Wi-Fi localization...")
    svm_wifi_model = SVC(kernel='linear', random_state=42)
    svm_wifi_model.fit(X_train_scaled, y_train)
    
    # 保存模型和标准化器
    joblib.dump(svm_wifi_model, 'svm_wifi_localization_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print(f"Training accuracy: {svm_wifi_model.score(X_test_scaled, y_test):.2f}")
    return svm_wifi_model, scaler

# 语音控制系统
def listen_to_user_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening for commands...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # 调整背景噪声
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f"Recognized command: {command}")  # 打印识别到的命令
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I could not understand the command.")
        return None
    except sr.RequestError as e:
        print(f"Recognition service error: {e}")
        return None

# 基于 Wi-Fi 信号预测房间
def control_lights_based_on_room(new_room, current_room, lights):
    room_mapping = {
        1: "Bedroom",
        
    }
    new_room_name = room_mapping.get(new_room, None)
    current_room_name = room_mapping.get(current_room, None)

    if new_room_name and new_room_name in lights and new_room_name != current_room_name:
        if current_room_name is not None and lights[current_room_name] == 1:
            lights[current_room_name] = 0
            print(f"User left room {current_room_name}. Turning off the light.")

        if lights[new_room_name] == 0:
            lights[new_room_name] = 1
            print(f"User entered room {new_room_name}. Turning on the light.")
    return lights


# 控制灯光亮度（增加亮度、降低亮度）
def adjust_light_brightness(room, brightness, lights_brightness):
    if brightness == 'increase':
        lights_brightness[room] = min(lights_brightness[room] + 1, 3)  # 最大亮度为3
        print(f"Brightness increased in room {room}. Current brightness level: {lights_brightness[room]}")
    elif brightness == 'decrease':
        lights_brightness[room] = max(lights_brightness[room] - 1, 0)  # 最低亮度为0
        print(f"Brightness decreased in room {room}. Current brightness level: {lights_brightness[room]}")

# 控制特定房间的灯光开关
file_path = r"C:\Users\H\html_templates\bedroom.html"
def control_light(state, room, lights, lights_brightness):
    if state == "on":
        lights[room] = 1
        print(f"Turning on the light in {room}.")
        # 打开并更新文件
        with open(file_path, "r+", encoding="utf-8") as file:
            content = file.read()
            updated_content = content.replace('<span id="lightStatus">Off</span>', '<span id="lightStatus">On</span>')
            file.seek(0)
            file.write(updated_content)
            file.truncate()
    elif state == "off":
        lights[room] = 0
        print(f"Turning off the light in {room}.")
        # 打开并更新文件
        with open(file_path, "r+", encoding="utf-8") as file:
            content = file.read()
            updated_content = content.replace('<span id="lightStatus">On</span>', '<span id="lightStatus">Off</span>')
            file.seek(0)
            file.write(updated_content)
            file.truncate()

def control_curtain(state, room):
    if state == 'open':
        print(f"Opening the curtain in room {room} via voice command.")
        # 如果需要修改 HTML 文件状态，参考以下代码：
        file_path = r"C:\Users\H\html_templates\bedroom.html"  # 修改为实际路径
        with open(file_path, "r+", encoding="utf-8") as file:
            content = file.read()
            updated_content = content.replace(
                '<span id="leftCurtainStatus">Closed</span>',
                '<span id="leftCurtainStatus">Open</span>'
            )
            updated_content = updated_content.replace(
                '<span id="rightCurtainStatus">Closed</span>',
                '<span id="rightCurtainStatus">Open</span>'
            )
            file.seek(0)
            file.write(updated_content)
            file.truncate()
    elif state == 'close':
        print(f"Closing the curtain in room {room} via voice command.")
        file_path = r"C:\Users\H\html_templates\bedroom.html"  # 修改为实际路径
        with open(file_path, "r+", encoding="utf-8") as file:
            content = file.read()
            updated_content = content.replace(
                '<span id="leftCurtainStatus">Open</span>',
                '<span id="leftCurtainStatus">Closed</span>'
            )
            updated_content = updated_content.replace(
                '<span id="rightCurtainStatus">Open</span>',
                '<span id="rightCurtainStatus">Closed</span>'
            )
            file.seek(0)
            file.write(updated_content)
            file.truncate()



# 控制窗帘


# 控制空调
def control_ac(state, room):
    file_path = r"C:\Users\H\html_templates\bedroom.html"
    if state == 'on':
        print(f"Turning on the air conditioner in room {room}.")
        # 更新 HTML 文件的状态
        with open(file_path, "r+", encoding="utf-8") as file:
            content = file.read()
            updated_content = content.replace('<span id="acStatus">Off</span>', '<span id="acStatus">On</span>')
            file.seek(0)
            file.write(updated_content)
            file.truncate()
    elif state == 'off':
        print(f"Turning off the air conditioner in room {room}.")
        with open(file_path, "r+", encoding="utf-8") as file:
            content = file.read()
            updated_content = content.replace('<span id="acStatus">On</span>', '<span id="acStatus">Off</span>')
            file.seek(0)
            file.write(updated_content)
            file.truncate()


# 处理语音命令的功能
def process_command(command, lights, lights_brightness):
    if command == "turn on light bedroom":
        control_light("on", "Bedroom", lights, lights_brightness)
    elif command == "turn off light bedroom":
        control_light("off", "Bedroom", lights, lights_brightness)
    elif command == "open curtain bedroom":
        control_curtain("open", "Bedroom")
    elif command == "close curtain bedroom":
        control_curtain("close", "Bedroom")
    elif command == "turn on ac bedroom":
        control_ac("on", "Bedroom")
    elif command == "turn off ac bedroom":
        control_ac("off", "Bedroom")
    else:
        print("Command not recognized or not supported.")




# 主函数：运行智能家居系统，自动开关灯并支持语音控制
def run_smart_home_system():
    svm_wifi_model, scaler = load_wifi_model()  # 加载或训练 Wi-Fi 模型
    if svm_wifi_model is None or scaler is None:
        return  # 模型未能加载或训练时退出

    current_room = None  # 当前房间初始化为空
    lights_brightness = {"Bedroom": 2}  # 仅保留卧室
    lights = {"Bedroom": 0}  # 默认灯光状态为关闭


    while True:
        real_time_data = get_real_time_data()
        
        # Wi-Fi 定位预测房间
        new_room = predict_room(svm_wifi_model, scaler)
        print(f"Predicted room: {new_room}, Current room: {current_room}")  # 增加调试输出
        
        if new_room != current_room:
            print(f"User moved from room {current_room} to room {new_room}.")
            
            # 自动控制灯光：离开当前房间时关闭灯，进入新房间时打开灯
            lights = control_lights_based_on_room(new_room, current_room, lights)

            # 更新当前房间
            current_room = new_room

        # 处理语音命令
        command = listen_to_user_command()
        if command:
            process_command(command, lights, lights_brightness)

        # 等待 1 秒再进行下一次检测，响应更快
        time.sleep(1)

# 运行智能家居系统
if __name__ == "__main__":
    run_smart_home_system()



# In[ ]:





# In[ ]:




