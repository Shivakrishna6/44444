import tensorflow as tf
from tensorflow.lite.python import lite
from tensorflow.lite.support.metadata import metadata_schema_pb2
from tensorflow.lite.support.metadata import metadata as tflite_metadata

# Load the TFLite model
model_path = 'model_fine_tuned.tflite'

# Create a Metadata Writer
writer = tflite_metadata.MetadataWriter.create_from_model_file(model_path)

# Define input and output specifications
input_tensor = metadata_schema_pb2.TensorMetadata()
input_tensor.name = 'input_tensor'
input_tensor.content = metadata_schema_pb2.ContentMetadata()
input_tensor.content.content = 'This tensor expects a float32 input of shape [1, height, width, channels]'

output_tensor = metadata_schema_pb2.TensorMetadata()
output_tensor.name = 'output_tensor'
output_tensor.content = metadata_schema_pb2.ContentMetadata()
output_tensor.content.content = 'This tensor outputs a float32 of shape [1, number_of_classes]'

# Add input and output tensor specifications
writer.set_input_metadata(input_tensor)
writer.set_output_metadata(output_tensor)

# Add additional metadata if needed
writer.add_metadata('Author', 'Shivakrishna6')

# Write the metadata to the model
output_model_path = 'model_fine_tuned_with_metadata.tflite'
writer.save(output_model_path)

print('Metadata added successfully and saved to:', output_model_path)