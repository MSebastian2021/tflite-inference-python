# TensorFlow Lite Python image classification demo

The tflite model used in this inference was created using the <a href="https://www.tensorflow.org/lite/convert#cmdline">command line tflite converter</a>, with the following options:

```
--input_shapes=4,640,640,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=FLOAT \
--allow_custom_ops
```

This was generated using the saved model in the Diversity repo.

# Resources
https://www.tensorflow.org/lite/guide/inference
https://github.com/JerryKurata/TFlite-object-detection
