# Use lightweight Python image
FROM python:3.10-slim

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all files
COPY . .

# Expose port
EXPOSE 7860

# Run the app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
