FROM python:latest
COPY . .
# Install
RUN make install
# Run tests
CMD ["make", "data"]