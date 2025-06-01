import cv2
import numpy as np
import os
from itertools import combinations

CALIBRATION_FILE = 'calibration.txt'

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

def calibrate():
    print("Calibration mode activated.")
    print("Measure a known distance on the image and enter its length in centimeters.")
    while True:
        try:
            cm_x = float(input("Enter length in cm along X axis: "))
            cm_y = float(input("Enter length in cm along Y axis: "))
            if cm_x > 0 and cm_y > 0:
                break
            else:
                print("Please enter positive numbers.")
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    save_calibration(cm_x, cm_y)
    print(f"Calibration saved: {cm_x} cm per width, {cm_y} cm per height")
    return cm_x, cm_y

def main():
    cm_per_pixel = load_calibration()
    if cm_per_pixel is None:
        cm_per_pixel = calibrate()
    cm_per_pixel_x, cm_per_pixel_y = cm_per_pixel

    # Пути к изображениям
    image1_path = 'i1_1.jpg'
    image2_path = 'i1_2.jpg'

    # Загрузка изображений
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print("Error loading images.")
        return
    else:
        print("Images loaded successfully.")

    # Применение Гауссовского размытия
    blurred_img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    blurred_img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    # Разница изображений
    diff = cv2.absdiff(blurred_img1, blurred_img2)
    _, thresh = cv2.threshold(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    centroid_combinations = list(combinations(centroids, 2))

    height, width, _ = diff.shape
    x_middle = width // 2
    vertex_bottom = (x_middle, height)
    vertex_left = (x_middle - 25, height - 50)
    vertex_right = (x_middle + 25, height - 50)
    pts = np.array([vertex_bottom, vertex_left, vertex_right], np.int32).reshape((-1,1,2))

    # Рисуем треугольник
    cv2.fillPoly(diff, [pts], (0, 255, 255))
    cv2.line(diff, vertex_bottom, vertex_left, (0, 255, 0), 2)
    cv2.line(diff, vertex_left, vertex_right, (0, 255, 0), 2)
    cv2.line(diff, vertex_right, vertex_bottom, (0, 255, 0), 2)

    # Находим пару с минимальным расстоянием
    min_distance = None
    min_pair = None
    for pair in centroid_combinations:
        distance = np.linalg.norm(np.array(pair[0]) - np.array(pair[1]))
        if min_distance is None or distance < min_distance:
            min_distance = distance
            min_pair = pair

    if min_pair is not None:
        centroid1, centroid2 = min_pair
        midpoint = ((centroid1[0] + centroid2[0]) // 2, (centroid1[1] + centroid2[1]) // 2)
        cv2.circle(diff, midpoint, 10, (255, 0, 0), -1)
        cv2.line(diff, centroid1, centroid2, (0, 255, 0), 2)
        cv2.putText(diff, 'STEP1', midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Второй отрезок и точка на 1/3 длины
        remaining_centroids = [c for c in centroids if c not in min_pair]
        if not remaining_centroids:
            print("Not enough centroids for further calculations.")
            cv2.imshow('Result', diff)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        nearest_unused_centroid = min(remaining_centroids, key=lambda c: np.linalg.norm(np.array(midpoint) - np.array(c)))
        cv2.line(diff, midpoint, nearest_unused_centroid, (0, 255, 0), 2)
        point_on_line2 = (int((2 * midpoint[0] + nearest_unused_centroid[0]) / 3), int((2 * midpoint[1] + nearest_unused_centroid[1]) / 3))
        cv2.circle(diff, point_on_line2, 10, (255, 0, 0), -1)
        cv2.putText(diff, 'STEP2', point_on_line2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        remaining_centroids2 = [c for c in centroids if c not in list(min_pair) + [nearest_unused_centroid]]
        if not remaining_centroids2:
            print("Not enough centroids for final calculation.")
            cv2.imshow('Result', diff)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        next_nearest_unused_centroid = min(remaining_centroids2, key=lambda c: np.linalg.norm(np.array(point_on_line2) - np.array(c)))
        cv2.line(diff, point_on_line2, next_nearest_unused_centroid, (0, 255, 0), 2)
        point_on_line4 = (int((3 * point_on_line2[0] + next_nearest_unused_centroid[0]) / 4), int((3 * point_on_line2[1] + next_nearest_unused_centroid[1]) / 4))
        cv2.circle(diff, point_on_line4, 10, (255, 0, 0), -1)
        cv2.putText(diff, 'STP', point_on_line4, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)

        # Линия отклонения
        cv2.line(diff, point_on_line4, vertex_bottom, (0, 255, 255), 8)
        otklonenie_x = (vertex_bottom[0] - point_on_line4[0]) * cm_per_pixel_x
        otklonenie_y = (vertex_bottom[1] - point_on_line4[1]) * cm_per_pixel_y

        print(f"STP deviation X (cm): {otklonenie_x:.2f}")
        print(f"STP deviation Y (cm): {otklonenie_y:.2f}")

        # Выводим значения отклонения на изображении — только латиница
        offset_x = 500
        offset_y = 50
        cv2.putText(diff, f'DEV_X_cm: {otklonenie_x:.2f}', 
                    (width - offset_x, height - offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(diff, f'DEV_Y_cm: {otklonenie_y:.2f}', 
                    (width - offset_x, height - offset_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Result', diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No close centroid pairs found.")

if __name__ == '__main__':
    main()
