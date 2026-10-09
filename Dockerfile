# 1. Standard lightweight Python core runtime engine base image environment
FROM python:3.10-slim

# 2. Establish isolated internal folder working parameters
WORKDIR /app

# 3. Pull deployment requirements manifest indexes
COPY requirements.txt .

# 4. Ingest and trigger library dependency building structures 
RUN pip install --no-cache-dir -r requirements.txt

# 5. Pull remaining working layout folders out to disk
COPY . .

# 6. Expose local port execution boundaries for network interfaces
EXPOSE 8501

# 7. Start the unified script interface wrapper engine
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
