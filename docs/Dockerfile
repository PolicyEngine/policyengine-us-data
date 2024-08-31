FROM python:latest
COPY . .
# Install
RUN make download
RUN make install
RUN make data
RUN make test
RUN make upload