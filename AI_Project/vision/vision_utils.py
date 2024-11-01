import cv2

def read_image(file_path):
    return cv2.imread(file_path)

def display_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
