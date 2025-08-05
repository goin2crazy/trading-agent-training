# 🐍 Base Python image
FROM python:3.10.11-slim

# 🌸 Set working directory inside container
WORKDIR /app

# 🧊 Install system dependencies (optional, extend as needed)
RUN apt-get update && apt-get install -y \
    git 

# 📦 Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🧁 Copy the rest of your app
COPY . .

# 🌈 Default command to run server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
