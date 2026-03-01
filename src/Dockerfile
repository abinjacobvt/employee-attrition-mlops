FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into system environment
RUN uv sync --frozen --no-dev

# Copy source code
COPY src ./src
COPY scripts ./scripts

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Run app using python -m (safer)
CMD ["python", "-m", "uvicorn", "attrition.api.main:app", "--host", "0.0.0.0", "--port", "8000"]