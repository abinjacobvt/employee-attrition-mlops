FROM python:3.11-slim

WORKDIR /app

# Copy entire project
COPY . .

# Install project dependencies
RUN pip install --no-cache-dir .

# Set python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["python", "-m", "uvicorn", "attrition.api.main:app", "--host", "0.0.0.0", "--port", "8000"]