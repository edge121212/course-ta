# 智慧課程助教系統 — Hugging Face Spaces (Docker SDK)
FROM python:3.12-slim

# 系統相依（部分套件編譯/執行需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 以非 root 使用者執行（HF Spaces 慣例，確保工作目錄可寫）
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# 先裝 CPU 版 torch，避免拉進龐大的 CUDA 套件（縮小映像、加快建置）
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

# 安裝其餘相依
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 複製程式碼
COPY --chown=user . .

EXPOSE 8501

# HF Spaces 會把 app_port(8501) 對外代理
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
