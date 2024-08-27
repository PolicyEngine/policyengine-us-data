FROM python:latest
COPY . .
# Install
RUN make install
RUN make download
RUN make data
RUN make test
