import os
import tensorflow as tf
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb


class ModelSaver:

    def __init__(self, model, description, author, license, version) -> None:

        self.model = model
        self.description = description
        self.author = author
        self.license = license
        self.version = version

        self.model_meta = None
        self.input_meta = None
        self.output_meta = None
        self.subgraph = None
        self.metadata_buf = None

    def __create_model_info__(self):
        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = self.model.name
        model_meta.description = (self.description)
        model_meta.version = self.version
        model_meta.author = self.author
        model_meta.license = (self.license)

        self.model_meta = model_meta

    def __create_input_metadata__(self, name, desc):
        input_meta = _metadata_fb.TensorMetadataT()

        input_meta.name = name
        input_meta.description = (desc.format(160, 160))
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
        input_meta.content.contentPropertiesType = (
            _metadata_fb.ContentProperties.FeatureProperties)

        self.input_meta = input_meta

    def __create_output_metadata__(self, labels_path):
        output_meta = _metadata_fb.TensorMetadataT()
        output_meta.name = "Probability"
        output_meta.description = "Probabilities of the 7 labels respectively."
        output_meta.content = _metadata_fb.ContentT()
        output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
        output_meta.content.contentPropertiesType = (
            _metadata_fb.ContentProperties.FeatureProperties)
        output_stats = _metadata_fb.StatsT()
        output_stats.max = [1.0]
        output_stats.min = [0.0]
        output_meta.stats = output_stats
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = os.path.basename(labels_path)
        label_file.description = "Labels for emotions that the model can recognize."
        label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
        output_meta.associatedFiles = [label_file]

        self.output_meta = output_meta

    def __create_subgraph__(self):
        self.subgraph = _metadata_fb.SubGraphMetadataT()

        if self.input_meta is None or self.output_meta is None or self.model_meta is None:
            print('input_meta in None or output_meta is None or model_meta is None')
            return

        self.subgraph.inputTensorMetadata = [self.input_meta]
        self.subgraph.outputTensorMetadata = [self.output_meta]
        self.model_meta.subgraphMetadata = [self.subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(
            self.model_meta.Pack(b),
            _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        self.metadata_buf = b.Output()

    def save_tflite(self, path, labels_path, i_name, i_desc, o_name, o_desc):

        print(f'save model {self.model.name} to {path}')
        self.__create_model_info__()
        self.__create_input_metadata__(i_name, i_desc)
        self.__create_output_metadata__(labels_path)
        self.__create_subgraph__()

        if self.metadata_buf is None:
            print('metadata_buf in None')
            return

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS,
                                               ]
        converter._experimental_lower_tensor_list_ops = False

        tflite_model = converter.convert()

        with open(path, 'wb') as f:
            f.write(tflite_model)

        populator = _metadata.MetadataPopulator.with_model_file(path)

        populator.load_metadata_buffer(self.metadata_buf)
        populator.load_associated_files([labels_path])
        populator.populate()
        

    def save_keras(self, path):

        self.model.save(path)


# test_path = r'D:\Programming\nn\emotion_classification\ser_diploma\model\test_model.keras'
# model_path = r'D:\Programming\nn\emotion_classification\ser_diploma\model\widecnn-res-gru-mel-v6-no-aug-no-scale-16khz.keras'
# ok_path = r'D:\Programming\nn\emotion_classification\ser_diploma\model\saved_models\cnn-lstm-i64x64-p109k-oAe50-f079-v1.keras'
# new_model = tf.keras.models.load_model(model_path)
# print(new_model.name)

# saver = ModelSaver(new_model, 'test saver', 'Mykola Nosenko', "", "v1")

# saver.save_tflite(f'{new_model.name}.tflite', r'D:\Programming\nn\emotion_classification\ser_diploma\model\emotion_labels.txt', 'mel', 'mel 64x64',
#                   'probability', 'probability of 7 labels respectively')
