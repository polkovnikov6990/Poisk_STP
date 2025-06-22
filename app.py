from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, abort
import os
import json
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from itertools import combinations
import base64
import pandas as pd
from datetime import datetime
import shutil
import atexit

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 мегабайта, можно больше

CALIBRATION_FILE = 'calibration.txt'
WEAPONS_FILE = 'weapons.json'

def save_calibration(cm_per_pixel_x, cm_per_pixel_y):
    with open(CALIBRATION_FILE, 'w') as f:
        f.write(f'{cm_per_pixel_x}\n{cm_per_pixel_y}\n')

def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return None
    with open(CALIBRATION_FILE, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return None
        return float(lines[0].strip()), float(lines[1].strip())

def load_weapons():
    if not os.path.exists(WEAPONS_FILE):
        return []
    with open(WEAPONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_bullet_holes(img_before_path, img_after_path, cm_x, cm_y):
    img_before = cv2.imread(img_before_path)
    img_after = cv2.imread(img_after_path)
    if img_before is None or img_after is None:
        return None, "Ошибка загрузки изображений."

    if img_before.shape != img_after.shape:
        img_after = cv2.resize(img_after, (img_before.shape[1], img_before.shape[0]))

    # 1. Разница между изображениями
    diff = cv2.absdiff(img_before, img_after)

    # 2. Гауссово размытие
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)

    # 3. Перевод в ч/б для поиска кругов
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # 4. Калибровка: пикселей в 1 мм
    height, width = gray.shape
    pixel_per_mm_x = width / (cm_x * 10)
    pixel_per_mm_y = height / (cm_y * 10)
    pixel_per_mm = (pixel_per_mm_x + pixel_per_mm_y) / 2

    min_diameter_mm = 4
    max_diameter_mm = 10

    min_radius_px = int((min_diameter_mm / 2) * pixel_per_mm)
    max_radius_px = int((max_diameter_mm / 2) * pixel_per_mm)

    min_area = np.pi * (min_radius_px ** 2)
    max_area = np.pi * (max_radius_px ** 2)

    # 5. Поиск кругов
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=15,
        param1=50,
        param2=30,
        minRadius=min_radius_px,
        maxRadius=max_radius_px
    )

    bullet_centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Обводим круг на изображении
            cv2.circle(blurred, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Центр круга
            cv2.circle(blurred, (i[0], i[1]), 2, (0, 0, 255), 3)
            bullet_centers.append((i[0], i[1]))

    # 6. STP — средняя точка попадания (если 3 или 4 пробоины)
    stp = None
    if len(bullet_centers) >= 3:
        xs = [c[0] for c in bullet_centers]
        ys = [c[1] for c in bullet_centers]
        stp = (int(np.mean(xs)), int(np.mean(ys)))
        cv2.circle(blurred, stp, 10, (255, 0, 255), -1)
        cv2.putText(blurred, 'STP', (stp[0]+10, stp[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # 7. Сохраняем результат
    result_img_path = os.path.join(UPLOAD_FOLDER, 'result.jpg')
    cv2.imwrite(result_img_path, blurred)

    result = {
        'img_path': result_img_path,
        'bullet_centers': bullet_centers,
        'stp': stp
    }
    result['otklonenie_x'] = result.get('otklonenie_x', 0)
    result['otklonenie_y'] = result.get('otklonenie_y', 0)
    return result, None

def save_base64_image(data_url, filename):
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    with open(filename, 'wb') as f:
        f.write(data)

@app.route('/')
def index():
    return redirect(url_for('select_weapon'))

@app.route('/select_weapon', methods=['GET', 'POST'])
def select_weapon():
    weapons = load_weapons()
    if request.method == 'POST':
        weapon_id = request.form.get('weapon')
        weapon_number = request.form.get('weapon_number')
        session['weapon_id'] = weapon_id
        session['weapon_number'] = weapon_number
        return redirect(url_for('capture_before'))
    return render_template('select_weapon.html', weapons=weapons)

@app.route('/capture_before', methods=['GET', 'POST'])
def capture_before():
    current_shot = session.get('current_shot', 1)
    weapon_number = session.get('weapon_number', '')
    weapon_name = ''
    weapon_id = session.get('weapon_id')
    if weapon_id:
        weapons = load_weapons()
        weapon = next((w for w in weapons if str(w['id']) == str(weapon_id)), None)
        if weapon:
            weapon_name = weapon['short_name']

    if request.method == 'POST':
        img_before_b64 = request.form.get('image_before')
        if img_before_b64:
            path1 = os.path.join(UPLOAD_FOLDER, 'before.jpg')
            save_base64_image(img_before_b64, path1)
            session['before_img_path'] = path1
            # СРАЗУ редирект на capture_after
            return redirect(url_for('capture_after'))
        else:
            flash('Сделайте снимок!', 'danger')
    return render_template('capture_before.html',
                          current_shot=current_shot,
                          weapon_name=weapon_name,
                          weapon_number=weapon_number)

@app.route('/start_check', methods=['GET', 'POST'])
def start_check():
    # Сбросить сессию для новой серии стрельб
    session['results'] = []
    session['current_shot'] = 1
    session.pop('before_img_path', None)
    return redirect(url_for('select_weapon'))

@app.route('/capture_after', methods=['GET', 'POST'])
def capture_after():
    if 'current_shot' not in session:
        session['current_shot'] = 1
    calib = load_calibration()
    weapons = load_weapons()
    weapon = next((w for w in weapons if str(w['id']) == session.get('weapon_id')), None)
    if request.method == 'POST':
        img_after_b64 = request.form.get('image_after')
        if img_after_b64 and calib:
            after_path = os.path.join('static/uploads', 'after.jpg')
            save_base64_image(img_after_b64, after_path)
            # Если это первая стрельба — "ДО" уже есть, иначе берем "ПОСЛЕ" предыдущей
            if 'before_img_path' not in session:
                flash('Нет исходного изображения "ДО"!', 'danger')
                return redirect(url_for('capture_before'))
            return redirect(url_for('normalize_diff', norm_level=50))
    return render_template('capture_after.html', weapon=weapon)

@app.route('/calibration', methods=['GET', 'POST'])
def calibration():
    if request.method == 'POST':
        try:
            cm_x = float(request.form['cm_x'])
            cm_y = float(request.form['cm_y'])
            if cm_x > 0 and cm_y > 0:
                save_calibration(cm_x, cm_y)
                flash('Калибровка сохранена!', 'success')
            else:
                flash('Введите положительные значения.', 'danger')
        except Exception:
            flash('Ошибка ввода.', 'danger')
    calib = load_calibration()
    return render_template('calibration.html', calib=calib)

@app.route('/export_results')
def export_results():
    results = session.get('results', [])
    if not results:
        flash('Нет данных для экспорта!', 'warning')
        return redirect(url_for('start_check'))
    df = pd.DataFrame(results)
    filename = f'strelba_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    filepath = os.path.join('static', 'uploads', filename)
    df.to_excel(filepath, index=False)
    return send_file(filepath, as_attachment=True)

@app.route('/reset_counter', methods=['POST'])
def reset_counter():
    # Сбросить счетчик и очистить сессию
    session['current_shot'] = 1
    session.pop('results', None)
    session.pop('before_img_path', None)
    session.pop('weapon_id', None)
    session.pop('weapon_number', None)
    # Очистить папку upload
    folder = os.path.join('static', 'uploads')
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}: {e}')
    return redirect(url_for('select_weapon'))

def clear_upload_folder():
    folder = os.path.join('static', 'uploads')
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}: {e}')

atexit.register(clear_upload_folder)

@app.route('/repeat_sample', methods=['POST'])
def repeat_sample():
    # Не меняем weapon_id и weapon_number
    session['current_shot'] = session.get('current_shot', 1) + 1
    # before_img_path уже указывает на последнее "ПОСЛЕ"
    return redirect(url_for('capture_after'))

@app.route('/next_sample', methods=['POST'])
def next_sample():
    # Сохраняем before_img_path (оно уже указывает на последнее "ПОСЛЕ")
    session['current_shot'] = 1  # или увеличивай глобальный счетчик, если нужно
    # Очищаем данные о предыдущем оружии
    session.pop('weapon_id', None)
    session.pop('weapon_number', None)
    return redirect(url_for('select_weapon'))

@app.route('/some_path')
def some_view():
    abort(403)

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    norm_level = int(request.form.get('norm_level', 50))
    # Преобразуй norm_level в параметры нормализации (например, alpha/beta или clipLimit для CLAHE)
    # Пример для alpha/beta:
    alpha = 0
    beta = int(155 + norm_level)  # 155..255
    img = cv2.imread('static/uploads/before.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm_img = cv2.normalize(gray, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite('static/norm_img.jpg', norm_img)
    # ... далее детекция кругов на norm_img ...
    return render_template('result.html', norm_level=norm_level)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, 'original.jpg')
            file.save(filepath)
            return redirect(url_for('normalize', norm_level=50))
    return render_template('upload.html')

@app.route('/normalize', methods=['GET', 'POST'])
def normalize():
    norm_level = int(request.values.get('norm_level', 50))
    filepath = os.path.join(UPLOAD_FOLDER, 'original.jpg')
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # Преобразуем norm_level в порог (например, для бинаризации)
    _, norm_img = cv2.threshold(img, norm_level, 255, cv2.THRESH_BINARY)
    norm_path = os.path.join(UPLOAD_FOLDER, 'norm_img.jpg')
    cv2.imwrite(norm_path, norm_img)
    return render_template('normalize.html', norm_img='uploads/norm_img.jpg', norm_level=norm_level)

@app.route('/normalize_diff', methods=['GET', 'POST'])
def normalize_diff():
    norm_level = int(request.values.get('norm_level', 50))
    before_path = 'static/uploads/before.jpg'
    after_path = 'static/uploads/after.jpg'
    before = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
    after = cv2.imread(after_path, cv2.IMREAD_GRAYSCALE)

    if before is None or after is None:
        return "Ошибка: одно из изображений не загружено!"

    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))

    diff = cv2.absdiff(after, before)
    # Нормализация разностного изображения
    _, norm_img = cv2.threshold(diff, norm_level, 255, cv2.THRESH_BINARY)
    norm_path = 'static/uploads/norm_diff.jpg'
    cv2.imwrite(norm_path, norm_img)

    # Предобработка
    img = cv2.imread(norm_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 5)
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    pixels_per_cm = 40  # например, 40
    pixels_per_mm = pixels_per_cm / 10

    pixels_per_mm = pixels_per_cm / 10
    min_diameter_mm = 4
    max_diameter_mm = 10

    min_radius_px = int((min_diameter_mm / 2) * pixels_per_mm)
    max_radius_px = int((max_diameter_mm / 2) * pixels_per_mm)

    min_area = np.pi * (min_radius_px ** 2)
    max_area = np.pi * (max_radius_px ** 2)

    img_color = img.copy()

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if 0.7 < circularity < 1.2 and min_area < area < max_area:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(img_color, (int(x), int(y)), int(radius), (0,255,0), 2)
            cv2.circle(img_color, (int(x), int(y)), 2, (0,0,255), 3)

    return render_template(
        'normalize_diff.html', 
        norm_img='uploads/norm_diff.jpg', 
        norm_level=norm_level
    )

@app.route('/find_stp', methods=['POST'])
def find_stp():
    norm_level = int(request.form.get('norm_level', 50))
    # Анализируешь norm_diff.jpg, ищешь круги, СТП и т.д.
    # ...
    result, error = analyze_bullet_holes('static/uploads/before.jpg', 'static/uploads/after.jpg', *load_calibration())
    if error:
        return render_template('error.html', error=error)
    return render_template('result.html', result=result, norm_level=norm_level)

@app.route('/get_norm_img')
def get_norm_img():
    norm_level = int(request.args.get('norm_level', 50))
    before_path = 'static/uploads/before.jpg'
    after_path = 'static/uploads/after.jpg'
    before = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
    after = cv2.imread(after_path, cv2.IMREAD_GRAYSCALE)
    if before is None or after is None:
        # Вернуть заглушку или ошибку
        return "Ошибка: изображение не найдено", 404
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    diff = cv2.absdiff(after, before)
    _, norm_img = cv2.threshold(diff, norm_level, 255, cv2.THRESH_BINARY)
    temp_path = 'static/uploads/norm_diff_temp.jpg'
    cv2.imwrite(temp_path, norm_img)
    return send_file(temp_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)