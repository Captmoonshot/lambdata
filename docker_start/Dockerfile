FROM debian

### *Minimal* Python container
### So logging/io works reliably w/Docker
ENV PYTHONUNBUFFERED=1

### Basic Python dev dependencies
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install python3-pip curl nano -y && \
  pip3 install pandas && \
  pip3 install scikit-learn==0.20.2 && \
  pip3 install -i https://test.pypi.org/simple/ lambdata-captmoonshot && \
  python3 -c "import lambdata_captmoonshot; print('Success!')"
