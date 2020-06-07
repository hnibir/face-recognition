class MotionSensor:
    def __init__(self):
        print('Motion sensor initialized')

    def check_movement(self):
        print("Movement checked")

    def check_if_human(self):
        print('Check if the moving object is human')


if __name__ == '__main__':
    motion_sensor = MotionSensor()
    motion_sensor.check_movement()
    motion_sensor.check_if_human()