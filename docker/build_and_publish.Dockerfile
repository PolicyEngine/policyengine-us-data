FROM python:latest
COPY . .
RUN make install
RUN make download
RUN make data
RUN make test
RUN make upload
