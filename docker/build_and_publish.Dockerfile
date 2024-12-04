FROM python:latest
COPY . .
RUN make install
CMD make generate-and-upload
