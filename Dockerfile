FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for sqlite-vec
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY BUGFS_mcp_server.py .
COPY crawl_issues_With_diffs.py .
COPY BM25.py .
COPY embedding.py .

# Create data and logs directories
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV BUGFS_DATA_DIR=/app/data
ENV BUGFS_LOG_DIR=/app/logs

# Expose port 1026
EXPOSE 1026

# Run the MCP server
CMD ["python", "BUGFS_mcp_server.py", "--host", "0.0.0.0", "--port", "1026"]
