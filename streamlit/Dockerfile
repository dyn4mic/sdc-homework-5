FROM python:3.8

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt
COPY . .

EXPOSE 8502

CMD ["streamlit", "run", "main.py"]
