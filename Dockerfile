FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
ENV TF_CPP_MIN_LOG_LEVEL=3
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
