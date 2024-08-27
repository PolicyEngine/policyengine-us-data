FROM python:latest
COPY . .
# Install
RUN make install
RUN ["make", "data"]
