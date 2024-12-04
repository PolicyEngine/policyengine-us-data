FROM python:3.11
COPY . .
RUN make install
CMD make generate-and-upload
