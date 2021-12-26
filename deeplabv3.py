import tensorflow as tf
from PIL.Image import Image
import numpy as np


class DeeplLabModel():
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        with tf.io.gfile.GFile(self.FROZEN_GRAPH_NAME, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="frozen_inference_graph")

        self.graph = graph

    def run(self,image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        input_layer = self.graph.get_tensor_by_name("frozen_inference_graph/"+self.INPUT_TENSOR_NAME)
        preds = self.graph.get_tensor_by_name("frozen_inference_graph/"+self.OUTPUT_TENSOR_NAME)
        pred_index = tf.argmax(preds)

        with tf.compat.v1.Session(graph=self.graph) as sess:
            class_labels, probs = sess.run([pred_index,preds],
                                           feed_dict={input_layer: [np.asarray(resized_image)]})

            return class_labels, probs[class_labels]


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

