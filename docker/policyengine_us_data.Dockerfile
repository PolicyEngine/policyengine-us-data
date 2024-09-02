FROM python:latest
COPY . .
RUN make install
