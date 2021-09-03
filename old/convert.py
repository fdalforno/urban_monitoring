import tensorflow as tf
import argparse


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    layers = [op.name for op in import_graph.get_operations()]

    if print_graph:
        print("-" * 50)
        print("Frozen model layers: ")

        for layer in layers:
            print(layer)
        
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


parser = argparse.ArgumentParser(description="Feature Extractor Model Converter")
parser.add_argument("--model",default="./models/mars-small128.pb",help="Path to freezed inference graph protobuf.")
parser.add_argument("--input",default="images:0",help="input tensor name")
parser.add_argument("--output",default="features:0",help="output tensor name")
parser.add_argument("--result",default="./models/mars-small128.tflite",help="output tflite model")
args = parser.parse_args()


with tf.io.gfile.GFile(args.model, "rb") as file_handle:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(file_handle.read())

print("Caricato grafo Tensorflow 1.X")

frozen_func = wrap_frozen_graph(graph_def=graph_def,inputs=args.input,outputs=args.output,print_graph=True)

print("Convertito grafo in Tensorflow 2.X")

converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()

with open(args.result, 'wb') as f:
  f.write(tflite_model)