FROM python:latest
COPY . .
RUN make install
RUN make download
RUN python docs/download.py
EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "docs/Home.py", "--server.port=8080", "--server.address=0.0.0.0"]
