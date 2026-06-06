# MIDI Generator frontend

This is a Vite/React static frontend. It can be deployed to GitHub Pages and configured to call a hosted model API for generation.

## Local development

```sh
npm ci
npm run dev
```

The dev server proxies `/api/*` to `http://localhost:5000`, so you can run the Flask backend locally and leave `VITE_API_URL` blank.

## Production configuration

Copy the example env file when building locally:

```sh
cp .env.example .env.local
```

Set:

- `VITE_API_URL` to the hosted model API origin, without `/api/generate`.
  - Example: `https://your-username-your-space.hf.space`
- `VITE_BASE_PATH` to the GitHub Pages base path.
  - User/org site: `/`
  - Project site: `/repo-name/`

The generation request is sent to:

```text
${VITE_API_URL}/api/generate
```

## GitHub Pages

The repository workflow at `.github/workflows/deploy-frontend.yml` builds this directory and deploys `dist/` to GitHub Pages.

In GitHub, set a repository variable named `VITE_API_URL` to the public HTTPS origin for the model API. The workflow automatically sets the correct Pages base path for both `username.github.io` repositories and project pages.

## Hosted model API requirements

The frontend expects the hosted API to expose:

- `GET /api/health`
- `POST /api/generate`

The request body is:

```json
{
  "notes": [{ "pitch": 60, "velocity": 100, "startTime": 0, "duration": 0.5 }],
  "tempo": 120,
  "genre": "jazz",
  "lengthMeasures": 4,
  "startMeasure": 2
}
```

The response body should include:

```json
{
  "notes": [{ "pitch": 64, "velocity": 100, "startTime": 2, "duration": 0.5 }]
}
```
