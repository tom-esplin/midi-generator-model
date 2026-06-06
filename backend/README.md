# Model API

The Flask API exposes the endpoints used by the static frontend:

- `GET /api/health`
- `POST /api/generate`

## Local development

```sh
pip install -r backend/requirements.txt
python backend/app.py
```

The local server listens on port `5000` by default. Set `PORT` to override it.

## Docker or Spaces deployment

The repository root includes a `Dockerfile` for hosting this API on Docker-based services such as Hugging Face Spaces free CPU tier:

```sh
docker build -t midi-generator-api .
docker run --rm -p 7860:7860 midi-generator-api
```

When deployed, set the frontend's `VITE_API_URL` to the public HTTPS origin of this API, for example:

```text
https://your-username-your-space.hf.space
```

The API image expects the trained artifacts to be present at:

- `models/model_weights/*.pt`
- `tokenization/saved_tokens/**/tokenizer.json`
