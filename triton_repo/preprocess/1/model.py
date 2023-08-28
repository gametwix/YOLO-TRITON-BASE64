
import triton_python_backend_utils as pb_utils
import numpy as np
import cv2
import base64

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for i, request in enumerate(requests):
            images_input = \
                pb_utils.get_input_tensor_by_name(request, "images_base64").as_numpy()
            decoded_images = []
            shapes = []
            for image in images_input:
                image_base64 = image[0].decode()
                
                image_bytes = base64.b64decode(image_base64)
                image_array = np.frombuffer(image_bytes, np.uint8)
                decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                resized_image = cv2.resize(decoded_image, (640, 640))
                resized_image = resized_image.astype(np.float32) / 255.0
                resized_image = np.transpose(resized_image, (2, 0, 1))
                decoded_images.append(resized_image)
                shapes.append(decoded_image.shape[:2])
            image_tensor = pb_utils.Tensor("images", np.array(decoded_images))
            shape_tensor = pb_utils.Tensor("shapes", np.array(shapes, dtype=np.int32))
            response = pb_utils.InferenceResponse(output_tensors=[image_tensor, shape_tensor])
            responses.append(response)
        return responses