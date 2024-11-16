
        #следующий этап - найти случайно отклонившуюся точку (>2,5 R) и научить в таком случае определять СТП по трем точкам
        #следующий этап - научить два изображения накладывать точно друг на друга
        ###и еще продумать очередность поступления фотографий в вычисления


import cv2
import numpy as np
import os
from itertools import combinations

# Пути к изображениям
image1_path = 'i3.jpg'
image2_path = 'i2.jpg'

# Загрузка изображений
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# Применение Гауссовского размытия к обоим изображениям
blurred_img1 = cv2.GaussianBlur(img1, (5, 5), 0)  # Параметры (5, 5) - размер ядра фильтра
blurred_img2 = cv2.GaussianBlur(img2, (5, 5), 0)


if blurred_img1 is None or blurred_img2 is None:
    print("Ошибка загрузки изображений.")
else:
    print("Изображения загружены успешно.")

    # Находим различия между изображениями
    diff = cv2.absdiff(blurred_img1, blurred_img2)

    # Применяем пороговую обработку для выделения различий
    _, thresh = cv2.threshold(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)

    # Находим контуры объектов на изображении с различиями
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем список для хранения центроидов облаков точек
    centroids = []

    # Находим центроиды для облаков точек
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    # Находим все возможные комбинации центроидов
    centroid_combinations = list(combinations(centroids, 2))

    # Находим значение контрольной точки КТ и наносим ее на результирующее изображение
    height, width, _ = diff.shape
    x_middle = width // 2
    y_zero = 0
    # Координаты вершин треугольника
    vertex_bottom = (x_middle, height)  # Вершина треугольника внизу
    vertex_left = (x_middle - 25, height - 50)  # Левая вершина
    vertex_right = (x_middle + 25, height - 50)  # Правая вершина
    # Задаем вершины треугольника
    pts = np.array([vertex_bottom, vertex_left, vertex_right], np.int32)
    pts = pts.reshape((-1,1,2))

    # Рисуем и заполняем треугольник желтым цветом
    cv2.fillPoly(diff, [pts], (0, 255, 255))

    # Рисуем треугольник
    cv2.line(diff, vertex_bottom, vertex_left, (0, 255, 0), 2)
    cv2.line(diff, vertex_left, vertex_right, (0, 255, 0), 2)
    cv2.line(diff, vertex_right, vertex_bottom, (0, 255, 0), 2)

    # Находим минимальное расстояние между центроидами
    min_distance = None
    min_pair = None
    for pair in centroid_combinations:
        distance = np.linalg.norm(np.array(pair[0]) - np.array(pair[1]))
        if min_distance is None or distance < min_distance:
            min_distance = distance
            min_pair = pair

    # Проверяем, что были найдены центроиды для ближайших облака точек
    if min_pair is not None:
        # Находим центры для ближайших облаков точек
        centroid1, centroid2 = min_pair

        # Находим середину отрезка, соединяющего центры ближайших облаков точек
        midpoint = ((centroid1[0] + centroid2[0]) // 2, (centroid1[1] + centroid2[1]) // 2)
        cv2.circle(diff, midpoint, 10, (255, 0, 0), -1)
        # Рисуем первый отрезок и надпись "SHAG1"
        cv2.line(diff, centroid1, centroid2, (0, 255, 0), 2)
        cv2.putText(diff, 'SHAG1', midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        #РИСУЕМ ВТОРОЙ ОТРЕЗОК И ИЩЕМ ТОЧКУ
        nearest_unused_centroid = min([c for c in centroids if c not in min_pair], key=lambda c: np.linalg.norm(np.array(midpoint) - np.array(c)))
        cv2.line(diff, midpoint, nearest_unused_centroid, (0, 255, 0), 2)
        
        # Находим точку на втором отрезке в первой трети
        point_on_line2 = (int((2 * midpoint[0] + nearest_unused_centroid[0]) / 3), int((2 * midpoint[1] + nearest_unused_centroid[1]) / 3))
        cv2.circle(diff, point_on_line2, 10, (255, 0, 0), -1)

        # Рисуем второй отрезок и надпись "SHAG2"
        cv2.putText(diff, 'SHAG2', point_on_line2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Находим следующий ближайший центроид, который не был использован
        next_nearest_unused_centroid = min([c for c in centroids if c not in list(min_pair) + [nearest_unused_centroid]], key=lambda c: np.linalg.norm(np.array(point_on_line2) - np.array(c)))

        # Рисуем отрезок от точки point_on_line2 до следующего ближайшего незадействованного центроида
        cv2.line(diff, point_on_line2, next_nearest_unused_centroid, (0, 255, 0), 2)

        # Находим точку, находящуюся на расстоянии 1/4 отрезка от point_on_line2 к next_nearest_unused_centroid
        point_on_line4 = (int((3 * point_on_line2[0] + next_nearest_unused_centroid[0]) / 4), int((3 * point_on_line2[1] + next_nearest_unused_centroid[1]) / 4))
        cv2.circle(diff, point_on_line4, 10, (255, 0, 0), -1)
        cv2.putText(diff, 'STP', point_on_line4, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)

        # РИСУЕМ ЛИНИЮ ОТКЛОНЕНИЯ СТП от ТП
        # Рисуем линию от точки point_line4 до вершины vertex_bottom
        cv2.line(diff, point_on_line4, vertex_bottom, (0, 255, 255), 8)
        otklonenie_x = vertex_bottom[0] - point_on_line4[0]
        otklonenie_y = vertex_bottom[1] - point_on_line4[1]
        print ("Отклонение СТП от ТП по оси Х в пикселях:", otklonenie_x)
        print ("Отклонение СТП от ТП по оси Y в пикселях:", otklonenie_y)

        # Выводим значения отклонений в правой нижней части экрана с использованием шрифта cv2.FONT_HERSHEY_TRIPLEX
        offset_x = 500
        offset_y = 50
        cv2.putText(diff, f'OTKLONENIE_X_px: {otklonenie_x}', 
                    (width - offset_x, height - offset_y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        cv2.putText(diff, f'OTKLONENIE_Y_px: {otklonenie_y}', 
                    (width - offset_x, height - offset_y + 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

        # Отображаем результат
        cv2.imshow('Image with Line to Nearest Unused Centroid', diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Не удалось найти ближайшие облака точек.")
