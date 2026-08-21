# Hywind Wind Intelligence — dashboard package

Self-contained. No models, no external data sources — everything the
app needs is in `data/`.

## Run directly

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py            # dev server on 0.0.0.0:8506
# production:
.venv/bin/gunicorn -w 2 -b 0.0.0.0:8506 app:server
```

## Or Docker

```
docker build -t hywind-app . && docker run -p 8506:8506 hywind-app
```

HOST/PORT env vars override the dev-server bind. Views:
`/?tab=data|models|operator`, `?theme=dark`.
