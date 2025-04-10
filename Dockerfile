FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/cache logs results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TIMEZONE=UTC

# Use production environment by default
RUN if [ -f .env.production ]; then cp .env.production .env; fi

# Expose port for monitoring (if needed)
EXPOSE 8080

# Run the trading agent in real-time mode
CMD ["python", "run_trading_agent_realtime.py"] 