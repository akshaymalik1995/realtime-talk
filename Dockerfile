FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_CONFIG=config/production.toml
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY config config
COPY prompts prompts
COPY assets/filler_marin.pcm assets/filler_marin.pcm

# run as a non-root user; the database lives on a mounted volume at /data (store.path in production.toml)
RUN useradd --create-home --uid 10001 desk && mkdir -p /data && chown desk /data
USER desk
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status != 200)"

# the key comes only from the environment: docker run -e OPENAI_API_KEY=... -v deskdata:/data -p 8000:8000 desk
CMD ["python", "-m", "app"]
