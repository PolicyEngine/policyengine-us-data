FROM python:latest
COPY . .
# Install
RUN make install
RUN make download
# Run tests
CMD ["make", "data"]