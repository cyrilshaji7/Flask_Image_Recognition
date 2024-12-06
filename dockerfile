# Add a line to make a pull request and add prof as reviewer
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Make port 9000 available
EXPOSE 9000

# Run the application
CMD ["python", "app.py"]