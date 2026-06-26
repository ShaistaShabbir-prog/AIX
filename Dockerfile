FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit

# Copy source
COPY . .

# Default command: run Streamlit demo
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

EXPOSE 8501
