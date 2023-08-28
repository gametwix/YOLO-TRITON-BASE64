
import triton_python_backend_utils as pb_utils
import numpy as np
import cv2
import json

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for i, request in enumerate(requests):
            answers = \
                pb_utils.get_input_tensor_by_name(request, "model_answer").as_numpy()
            shapes = \
                pb_utils.get_input_tensor_by_name(request, "shapes").as_numpy()
            jsons = []
            for answer, shape in zip(answers, shapes):
                outputs = np.array([cv2.transpose(answer)])
                rows = outputs.shape[1]

                boxes = []
                scores = []
                class_ids = []

                for i in range(rows):
                    classes_scores = outputs[0][i][4:]
                    (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                    if maxScore >= 0.4:
                        box = [
                            (outputs[0][i][0] - (0.5 * outputs[0][i][2]))/640*shape[1], (outputs[0][i][1] - (0.5 * outputs[0][i][3]))/640*shape[0],
                            outputs[0][i][2]/640*shape[1], outputs[0][i][3]/640*shape[0]]
                        boxes.append(box)
                        scores.append(maxScore)
                        class_ids.append(maxClassIndex)
                objects = []
                for bbox, score, class_id in zip(boxes, scores, class_ids):
                    obj = {"bbox":list(map(str, bbox)), "score":str(score), "class":str(class_id)}
                    objects.append(obj)
                objects_json = json.dumps(objects)
                jsons.append([objects_json])
            np_data = np.array(jsons, dtype=object)
            out_tensor = pb_utils.Tensor("output_json", np_data)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        return responses