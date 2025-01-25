import tensorflow as tf
import os


class ImageLabeling:
    def __init__(self, image_dataset_dir: str = None, labels: dict = None):
        self.image_dataset_dir = image_dataset_dir
        self.labels = labels
        self.dataset = None
        self.parsed_dataset = None

    def raise_error_if_none(self):
        if self.image_dataset_dir is None:
            raise ValueError(f"Expected image_dataset_dir to be different than {self.image_dataset_dir}")
        
        elif self.labels is None:
            raise ValueError(f"Expected labels to be different than {self.labels}")

    def record(self, tfrecord_file_name):
        self.raise_error_if_none()

        # function to create a TensorFlow feature from a bytes value (following tfrecord documentation)
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        # function to create a TensorFlow feature from an integer value (following tfrecord documentation)
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        # open TFRecord writer directly
        with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
            # Sort the files to ensure they are processed in the correct order
            image_files = sorted(os.listdir(self.image_dataset_dir))

            # loop through the image directory and create examples
            for image_name in image_files:
                image_path = os.path.join(self.image_dataset_dir, image_name)

        
                class_name = image_name.split('_')[0] # extract the class label based on the image name (e.g., "healthy_0".split('_')[0] = "healthy")
                label = self.labels[class_name]  # map the class to a numeric label based on the dictionary

                # read the image as binary and store it in the image_data variable
                with open(image_path, "rb") as f:
                    image_data = f.read()

                # create a TFRecord example
                feature = {
                    'image': _bytes_feature(image_data),  # Convert image data (bytes) into a format that TensorFlow can store
                    'label': _int64_feature(label),  # Convert the label into a format that TensorFlow can store
                }

                # Create an Example
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Write the serialized example to the file
                writer.write(example.SerializeToString())

        self.load_data(tfrecord_file_name)  # Loading and creating dataset variable
        print(f"TFRecord saved to {tfrecord_file_name}")

    def load_data(self, tfrecord_file_name):
        # Function to parse the TFRecord examples
        def parse_tfrecord(example_proto):
            # Define the expected features
            feature_description = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            }
            return tf.io.parse_single_example(example_proto, feature_description)

        # load TFRecord file
        self.dataset = tf.data.TFRecordDataset(tfrecord_file_name)

        # parse examples in the dataset
        self.parsed_dataset = self.dataset.map(parse_tfrecord)

        # check if dataset has examples
        try:
            sample = next(iter(self.parsed_dataset))
            print("Successfully loaded")

        except StopIteration:
            print("No data found in the parsed dataset.")
