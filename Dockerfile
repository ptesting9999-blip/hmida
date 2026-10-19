FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 1000 leadgen \
    && mkdir -p /app/data /app/output \
    && chown -R leadgen /app

USER leadgen
# data/ holds the SQLite state and suppression list; output/ holds the lead files.
VOLUME ["/app/data", "/app/output"]
ENTRYPOINT ["leadgen"]
CMD ["run", "--every", "6"]
