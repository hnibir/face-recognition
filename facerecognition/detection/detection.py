import mtcnn
from os import listdir
from os.path import isdir
from PIL import Image
import numpy as np


class FaceDetection:
    detector = None
    i = 1

    def __init__(self):
        print("MTCNN version: " + mtcnn.__version__)
        print("FaceDetection class initialized")
        # create the detector, using default weights
        self.detector = mtcnn.MTCNN()

    # extract a single face from a given photograph
    def extract_face(self, filename=None, required_size=(96, 96)):
        # load image from file
        image = Image.open(filename)
        # convert image to RGB if needed
        image = image.convert('RGB')
        # convert image to array
        pixels = np.asarray(image)
        # detect faces in the image
        results = self.detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        # print(filename.split(sep=".jpg"))
        image.save(filename.split(sep=".jpg")[0] + '_cropped' + '.jpg')
        return image

    # load images and extract faces for all images in a directory
    def load_faces(self, directory=None):
        faces = list()
        # enumerate files
        for filename in listdir(directory):
            path = directory + filename
            if 'cropped' in path:
                continue
            # get face
            face = self.extract_face(path)
            faces.append(face)
        return faces

    # load a dataset that contains one subdir for each class that in turn contains images‚
    def load_dataset(self, directory=None):
        print("Loading dataset")
        # enumerate folders on per class
        for subdir in listdir(directory):
            # subdirectory path
            subdirectory = directory + subdir + '/'
            # skip any other files that might be in the sub directory
            faces = self.load_faces(subdirectory)
            # summarize progress
            print('>Loaded %d examples for class: %s' % (len(faces), subdir))
        return faces


if __name__ == "__main__":
    fc = FaceDetection()
    fc.load_dataset('../../images/')
