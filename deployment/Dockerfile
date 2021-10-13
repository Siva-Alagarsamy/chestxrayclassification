FROM python:3.8-slim as builder
COPY . /src
RUN pip install --user flask tensorflow

FROM python:3.8-slim as app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /src .

ENV PATH=/root/.local:$PATH
EXPOSE 5000

CMD ["python3", "sentiment_rest_server.py"]