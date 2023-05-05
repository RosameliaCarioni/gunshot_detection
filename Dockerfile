FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /tf/notebooks/

# Copy the file with the requirements to the '/app' directory.
COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

EXPOSE 8888


