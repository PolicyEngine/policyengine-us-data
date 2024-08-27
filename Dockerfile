FROM python:latest
COPY . .
# Install
RUN make install
CMD ["make", "data"]
