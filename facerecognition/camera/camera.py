class Camera:
    def __init__(self):
        print('Camera initialized')

    def capture_picture(self):
        print("Picture captured")

    def save_picture(self):
        print('Picture saved')


if __name__ == '__main__':
    camera = Camera()
    camera.capture_picture()
    camera.save_picture()