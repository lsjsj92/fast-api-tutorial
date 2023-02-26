# Base image
FROM python:3.8

# Install pytorch
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Set working directory
WORKDIR /fastapi

# Install python lib
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# Copy files
COPY ./models /fastapi/models
COPY ./main.py /fastapi/
COPY ./packages /fastapi/packages
COPY ./example/train_model /fastapi/example/train_model

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8088"]
