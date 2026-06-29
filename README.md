# Space of Mind — brain-service

Python sidecar that takes a recorded audio clip and returns a brand-styled
cortical activation map plus the top brain regions, computed via Meta's
**TRIBE v2** model on the fsaverage5 surface mesh.

Called from the Next.js app's `/api/analyze` route at [lib/brain.ts](../lib/brain.ts).

---

## ⚠️ License caveat — read first

**TRIBE v2 is released under CC BY-NC.** That means:

- ✅ Internal research, internal demos, NY Tech Week-style booths with no
  commercial transaction tied to the brain map are fine.
- ❌ Selling access to the brain map, including it as a feature in a paid
  product, or using it in commercial marketing is not allowed without a
  separate license from Meta.

This service is built and documented for the research-use case. If Space
of Mind's commercial trajectory changes, the brain feature has to be
rebuilt against a different model.

---

## Pipeline

```
audio (.webm / .wav / .m4a)
  └─► WhisperX (runs inside tribev2.get_events_dataframe)
        └─► word-level event DataFrame
              └─► TribeModel.predict(events=df)
                    └─► (T, V) cortical activation tensor on fsaverage5
                          ├─► pick_peak_timestep   ─► (V,) peak vector
                          ├─► Destrieux region decoder ─► top 4 regions
                          └─► nilearn surface plot (brand palette) ─► PNG
```

Output: brand-styled PNG (base64), top region list with citations, the WhisperX transcript text, the peak timestep index.

---

## Local development

This service is heavy (PyTorch + tribev2 + nilearn + fsaverage5 mesh).
Local dev only really works on a workstation with ≥ 8 GB free RAM.

```bash
cd brain-service

# 1. HuggingFace login (one-time per machine)
#    Get a read token at https://huggingface.co/settings/tokens
#    AND apply for access to facebook/tribev2 first — without
#    approval, the download will 401.
huggingface-cli login

# 2. Install deps. PyTorch wheels are big — first run is slow.
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 torchaudio==2.5.1
pip install -r requirements.txt
pip install --no-deps "git+https://github.com/facebookresearch/tribev2.git@main"

# 3. Run the service
uvicorn app:app --reload --host 0.0.0.0 --port 8080

# 4. Smoke-test
curl http://localhost:8080/health
# {"ok":true,"service":"brain-service"}

# 5. Warm the model (first request is ~30s)
curl -X POST http://localhost:8080/warmup

# 6. Render a brain map from a local audio file
curl -X POST http://localhost:8080/render \
     -F "audio=@/path/to/recording.webm" \
     | jq .
```

---

## Deploying to RunPod Serverless

### One-time setup

1. **Get TRIBE access on HuggingFace.** Apply at
   https://huggingface.co/facebook/tribev2 — Meta approves manually,
   typically within a day or two for research use. Repeat for
   https://huggingface.co/meta-llama/Llama-3.2-3B (used as TRIBE's text
   encoder). Create a read-only token at
   https://huggingface.co/settings/tokens.

2. **Build the Docker image** and push to a registry RunPod can pull from
   (Docker Hub, GitHub Container Registry, or RunPod's own image upload).
   From `brain-service/`:

   ```powershell
   # Build for GPU (CUDA wheels)
   docker build --build-arg COMPUTE=gpu -t spaceofmind-brain:latest .

   # Tag + push to Docker Hub (example)
   docker tag spaceofmind-brain:latest <your-dockerhub>/spaceofmind-brain:latest
   docker push <your-dockerhub>/spaceofmind-brain:latest
   ```

3. **Create a RunPod Network Volume** for model caches so workers don't
   re-download multi-GB weights on every cold start. RunPod dashboard →
   Storage → Network Volumes → New. Size: **50 GB** minimum.

4. **Create a RunPod Serverless Endpoint.** RunPod dashboard → Serverless
   → New Endpoint:
   - **Image**: your pushed image (e.g. `docker.io/<your-dockerhub>/spaceofmind-brain:latest`)
   - **GPU type**: pick a 24 GB GPU (L4, A4000, A5000, or RTX 4090 if available)
   - **Container disk**: at least 20 GB
   - **Network volume**: attach the one from step 3, mount at `/app/cache`
   - **Max workers**: **1** for a single iPad booth (2 only if you truly need concurrent renders)
   - **Active workers**: **0** (flex — scale to zero; pennies per render, $0 overnight)
   - **Idle timeout**: **300 sec** (keeps worker warm between booth visitors during an event)
   - **FlashBoot**: enabled (Standard)
   - **Execution timeout**: **600 sec**
   - **Container start command**: leave empty (Dockerfile CMD runs `handler.py`)
   - **Environment variables**: set per the table below

### Environment variables (set on the RunPod endpoint)

| Variable | Required | Notes |
|---|---|---|
| `HF_TOKEN` | ✅ yes | HuggingFace token with access to `facebook/tribev2` AND `meta-llama/Llama-3.2-3B`. |
| `TRIBE_MODEL_ID` | optional | Defaults to `facebook/tribev2`. Override only if Meta publishes a successor. |
| `TOP_K_REGIONS` | optional | Defaults to `4`. How many brain regions to surface per render. |
| `OUTPUT_DPI` | optional | Defaults to `180`. PNG render resolution. |

`SERVICE_AUTH_TOKEN` is no longer needed — RunPod's API key auth replaces it.

### Wiring back to Vercel

After your endpoint is live, RunPod will give you an Endpoint ID
(e.g. `abc123xyz`). Add these to your Vercel project:

```
RUNPOD_ENDPOINT_ID=abc123xyz
RUNPOD_API_KEY=<your RunPod API key from Settings → API Keys>
```

The Vercel app reads these in [lib/brain.ts](../lib/brain.ts). If either
is missing, the analyze pipeline skips brain render and continues — the
rest of the analysis still ships.

### First call after deploy

The first call to a fresh endpoint triggers:
- Worker cold start (~30s container spin-up)
- TRIBE model download from HF + load to GPU (~1–2 min on cold network volume)
- WhisperX environment setup via uvx (~1 min)
- Llama-3.2-3B download + load to GPU (~2–4 min on cold network volume)

Once the network volume is warm with all the model files, subsequent
worker cold starts skip the downloads and only pay for VRAM load
(~30–60s total).

**Cheap warmup (no audio):** the NYTech booth calls this automatically at
Intake via `POST /api/brain/warmup`, which submits `{ "warmup_only": true }`
to RunPod `/run`. You can also smoke-test manually:

```bash
curl -X POST \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"warmup_only":true}}' \
  "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync"
```

Expect `{ "ok": true, "warmed": true }` once TRIBE + atlases are loaded.

**Full render smoke test** (optional, uses real inference):

```powershell
$body = @{
  input = @{
    audio_b64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes("assets/test.wav"))
    audio_format = "wav"
  }
} | ConvertTo-Json

curl -X POST `
     -H "Authorization: Bearer $env:RUNPOD_API_KEY" `
     -H "Content-Type: application/json" `
     -d $body `
     "https://api.runpod.ai/v2/$env:RUNPOD_ENDPOINT_ID/runsync"
```

First cold call with empty volume: ~5–10 min (downloads + load). Warm worker:
~30–90s per booth recording.

### Local dev still works

The old FastAPI `app.py` is preserved for local iteration:

```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

You can test the same code paths locally before pushing the image to
RunPod. The `handler.py` entry point is only invoked when the container
runs under RunPod's serverless runtime.

### Wiring back to Vercel

Add two env vars to your Vercel project:

```
BRAIN_SERVICE_URL=https://your-brain-service.up.railway.app
BRAIN_SERVICE_TOKEN=<same value as SERVICE_AUTH_TOKEN on Railway>
```

The Next.js app reads these in [lib/brain.ts](../lib/brain.ts). If the
URL is missing, the analyze pipeline skips the brain render and continues
without one — the rest of the analysis is unaffected.

---

## API

### `GET /health`
Liveness probe — returns 200 as soon as the process is up. Does NOT
indicate readiness. Don't use this for routing decisions.

### `GET /ready`
Readiness probe. Returns **503** while startup warmup is in progress and
**200** once TRIBE + atlases are loaded. Railway's healthcheck is wired
to this path so traffic only routes when the deploy can actually serve.

```jsonc
// 200 response
{
  "ready": true,
  "tribe_loaded": true,
  "atlases_loaded": true,
  "full_warmup_done": true,    // false until first /render succeeds
  "uptime_seconds": 47.3,
  "last_error": null
}
```

### `GET /warmup`
Manual force-warmup. Idempotent — re-runs missing startup steps. Useful
if `/ready` reported a partial-warmup failure you've since fixed
(e.g. `HF_TOKEN` updated after deploy). Gated by `SERVICE_AUTH_TOKEN`.

### `POST /render`
Multipart form with one file field `audio`. Optional auth header.

Returns `RenderOut`:
```jsonc
{
  "brain_image_base64": "iVBORw0KGgoAAAANSUhEU...",
  "top_regions": [
    {
      "id": "DMN_CORE",
      "scientific_name": "Default Mode Network",
      "anatomical_descriptor": "medial prefrontal cortex + posterior cingulate cortex",
      "yeo_network": "Default",
      "short_function": "self-referential thought; internal narration; autobiographical memory",
      "function_summary": "The Default Mode Network is the most-studied network...",
      "score": 0.42
    }
    // ...
  ],
  "dominant_yeo_network": "Default",
  "transcript_text": "I think I'm closing my seed round in about eight weeks...",
  "peak_timestep": 23
}
```

---

## File map

```
brain-service/
├── Dockerfile          Build image (CPU default, GPU via COMPUTE=gpu)
├── railway.json        Railway build + healthcheck config
├── requirements.txt    PyPI deps (PyTorch installed separately in Dockerfile)
├── config.py           Env-driven config (cache paths, auth token, etc.)
├── inference.py        TRIBE singleton + run_tribe + pick_peak_timestep
├── regions.py          Destrieux atlas → app-level region decoder
├── render.py           Brand-styled nilearn surface plot
├── label_library.yaml  12 functional clusters with citations
├── app.py              FastAPI entry point — /health, /warmup, /render
└── README.md           This file
```

---

## Provenance

- **TRIBE v2**: Meta AI Research, https://huggingface.co/facebook/tribev2 — CC BY-NC.
- **fsaverage5 surface** + **Destrieux atlas**: FreeSurfer / nilearn, distributed under their respective licenses.
- **Label library** (`label_library.yaml`): adapted from sshandhra1/self-talk-mirror (regional groupings + scientific citations).
- **Brand-styled colormap + nilearn render**: original to this repo.
