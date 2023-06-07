FROM python:3.10-slim
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
WORKDIR /app
COPY . .
CMD streamlit run app.py --server.address=0.0.0.0 --server.port=8501
