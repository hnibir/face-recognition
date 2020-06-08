import tensorflow as tf
import cv2 as cv
from facerecognition.inception.inception_blocks_v2 import *
from facerecognition.utils.fr_utils import *
from keras.models import load_model
from os import listdir


class FaceRecognition:
    FRModel = None

    def __init__(self):
        print("FaceRecognition initialized")
        # tf.keras.backend.set_image_data_format('channels_first')
        # self.FRModel = load_model('../../datasets/facenet_keras.h5')
        self.FRModel = faceRecoModel(input_shape=(3, 96, 96))
        print("Total params: " + str(self.FRModel.count_params()))
        self.FRModel.compile(optimizer='adam', loss=self.triplet_loss, metrics=['accuracy'])
        load_weights_from_FaceNet(self.FRModel)

        # print(self.FRModel.summary())
        # print(self.FRModel.count_params())

    def triplet_loss(self, y_true, y_pred, alpha=0.2):
        """
        Implementation of the triplet loss as defined by formula (3)

        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor images, of shape (None, 128)
                positive -- the encodings for the positive images, of shape (None, 128)
                negative -- the encodings for the negative images, of shape (None, 128)

        Returns:
        loss -- real number, value of the loss
        """

        print("Calculate Triplet loss")

        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

        # Step 1: Compute the (encoding) distance between the anchor and the positive
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
        # Step 2: Compute the (encoding) distance between the anchor and the negative
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
        # Step 3: subtract the two previous distances and add alpha.
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
        return loss

    def triplet_loss2(self, y_true, y_pred, alpha=0.2):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=-1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=-1)
        return tf.maximum(positive_dist - negative_dist + alpha, 0.)

    """
    # get the face embedding for one face
    def get_embedding(self, image_path=None):
        img1 = cv.imread(image_path, 1)
        img = img1[..., ::-1]
        # img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12) # make (3, 160, 160)
        img = np.around(img / 255.0, decimals=12)
        x_train = np.array([img])

        embedding = self.FRModel.predict_on_batch(x_train)
        return embedding
    """

    # get the face embedding for one face
    def img_to_encoding(self, image_path=None):
        return img_to_encoding(image_path=image_path, model=self.FRModel)

    def verify(self, image_path, identity, database):
        """
        Function that verifies if the person on the "image_path" image is "identity".

        Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras

        Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
        """

        # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
        # encoding = self.get_embedding(image_path=image_path)
        encoding = self.img_to_encoding(image_path=image_path)

        # Step 2: Compute distance with identity's image (≈ 1 line)
        # dist = np.linalg.norm(encoding - database[identity])
        min_dist = 5000
        for encoding_from_db in database[identity]:
            dist = np.linalg.norm(encoding - encoding_from_db)
            if dist < min_dist:
                min_dist = dist

        # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
        if min_dist < 0.5:
            print("It's " + str(identity) + ", welcome in!" + " the distance is " + str(min_dist))
            door_open = True
        else:
            print("It's not " + str(identity) + ", please go away" + " the distance is " + str(min_dist))
            door_open = False

        return min_dist, door_open

    def who_is_it(self, image_path, database):
        """
        Implements face recognition for the office by finding who is the person on the image_path image.

        Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras

        Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
        """

        # Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
        # encoding = self.get_embedding(image_path=image_path)
        encoding = self.img_to_encoding(image_path=image_path)

        # Step 2: Find the closest encoding
        # Initialize "min_dist" to a large value, say 100 (≈1 line)
        min_dist = 5000

        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in database.items():
            for enc in db_enc:
                # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
                dist = np.linalg.norm(encoding - enc)
                # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
                # (≈ 3 lines)
                if dist < min_dist:
                    min_dist = dist
                    identity = name

        if min_dist > 0.5:
            print("Not in the database." + ", the distance is " + str(min_dist))
        else:
            print("It's " + str(identity) + ", the distance is " + str(min_dist))

        return min_dist, identity

    def get_database(self):
        database = dict()

        img_directory = '../../images/'
        for subdir in listdir(img_directory):
            database[subdir] = list()
            subsubdir = img_directory + subdir + '/'
            for filename in listdir(subsubdir):
                path = subsubdir + filename
                if 'cropped' in path:
                    encoding = fr.img_to_encoding(image_path=path)
                    database[subdir].append(encoding)
        return database


if __name__ == '__main__':
    fr = FaceRecognition()

    database = fr.get_database()
    # database["salma"] = fr.img_to_encoding(image_path="../../images/salma/salma_01_cropped.jpg")
    # database["sajib"] = fr.img_to_encoding(image_path="../../images/sajib/sajib_02_cropped.jpg")

    fr.verify(image_path="../../images/sajib/sajib_03_test.jpg", identity='sajib', database=database)
    fr.verify(image_path="../../images/salma/salma_02_test.jpg", identity='salma', database=database)

    fr.who_is_it(image_path="../../images/sajib/sajib_03_test.jpg", database=database)
