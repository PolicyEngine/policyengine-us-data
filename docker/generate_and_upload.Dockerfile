FROM python:latest
COPY . .
RUN make install
RUN make upload
