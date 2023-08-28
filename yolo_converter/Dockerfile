FROM nvcr.io/nvidia/tritonserver:22.12-py3-sdk

RUN pip install --no-cache-dir --upgrade \
    pip setuptools wheel nvidia-pyindex 
    
RUN pip install --no-cache-dir --upgrade \
    tensorrt==8.5.1.7 \
    triton-model-navigator==0.5.6 \
    ultralytics==8.0.124

RUN sed -i '55,59d;' /usr/local/lib/python3.8/dist-packages/ultralytics/nn/modules/head.py && \
    sed -i '55i\\        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)' \
    /usr/local/lib/python3.8/dist-packages/ultralytics/nn/modules/head.py && \
    sed -i '56,58d' /usr/local/lib/python3.8/dist-packages/ultralytics/nn/autoshape.py


WORKDIR /app

COPY ./navigator.py /app/


ENTRYPOINT ["python3", "navigator.py"]
