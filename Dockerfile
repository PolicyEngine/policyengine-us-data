FROM python:latest
COPY . .
# Install
RUN pip install -e .
# Run tests
CMD ["make", "test"]