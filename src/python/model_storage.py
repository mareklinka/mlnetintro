from keras.models import load_model as load_keras_model
from os.path import join
import pickle
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import keras as K
import tensorflow as tf

def save_model(model, filename):
    """
    Saves the specified Keras model into a file.

    Parameters
    ----------

    model : Keras model
        The model to store
    filename : string
        The file name to give to the resulting file
    """

    model.save(join("models", filename + ".h5"))

def load_model(filename):
    """
    Loads the specified Keras model from a file.

    Parameters
    ----------
    filename : string
        The name of the file to read from

    Returns
    -------
    Keras model
        The Keras model loaded from a file
    """

    return load_keras_model(join("models", filename + ".h5"))

def save_history(history, filename):
    """
    Saves the specified Keras model hostiory into a file.

    Parameters
    ----------

    model : Keras model
        The model history to store
    filename : string
        The file name to give to the resulting file
    """

    with open(join("models", filename + "_history.bin"), 'wb') as h:
        pickle.dump(history.history, h)

def load_history(filename):
    """
    Loads the model history from the specified file.

    Parameters
    ----------
    filename : string
        The name of the file to read from

    Returns
    -------
    Keras model
        The model history loaded from the file.
    """

    with open(join("models", filename + "_history.bin"), 'rb') as h: # saving the history of the model
        history = pickle.load(h)
        return history

def convert_to_tensorflow(filename):
    K.backend.set_learning_phase(0)

    output_names = ["final_layer/Sigmoid"]
    model = K.models.load_model(join("models", filename + ".h5"))
    session = K.backend.get_session()
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference([]))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        for node in input_graph_def.node:
            node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        tf.train.write_graph(frozen_graph, "models", filename + ".pb", as_text=False)