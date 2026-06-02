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

## Deploying to Railway

### One-time setup

1. **Get TRIBE access on HuggingFace.** Apply at
   https://huggingface.co/facebook/tribev2 — Meta approves manually,
   typically within a day or two for research use. Create a read-only
   token at https://huggingface.co/settings/tokens.
2. **Create a new Railway project** pointing at this `brain-service/`
   directory. Railway → New Project → Deploy from GitHub repo → choose
   this repo → set the **root directory** to `brain-service/`. Railway
   reads `Dockerfile` automatically.
3. **Attach a persistent volume** at `/app/cache` so cold redeploys don't
   re-download the TRIBE checkpoint, fsaverage5 mesh, and Destrieux atlas.
   Railway → Storage → New Volume → mount path `/app/cache`. Minimum
   recommended size: **15 GB** (TRIBE weights alone are several GB).

### Environment variables (set in Railway → Variables)

| Variable | Required | Notes |
|---|---|---|
| `HF_TOKEN` | ✅ yes | Your HuggingFace token with access to `facebook/tribev2`. tribev2 reads this on first import. |
| `SERVICE_AUTH_TOKEN` | recommended | Long random string. Vercel must send `Authorization: Bearer <this>` on every request. Defense-in-depth against drive-by traffic. |
| `TRIBE_MODEL_ID` | optional | Defaults to `facebook/tribev2`. Override only if Meta publishes a successor checkpoint. |
| `TOP_K_REGIONS` | optional | Defaults to `4`. How many brain regions to surface per recording. |
| `OUTPUT_DPI` | optional | Defaults to `180`. PNG render resolution. |
| `PORT` | auto | Set by Railway. Don't override. |

### CPU vs GPU

This image builds for CPU by default. To run on GPU:

```bash
# In Railway → Settings → Build → Build Args
COMPUTE=gpu
```

You'll also need to provision a Railway GPU host (Settings → Compute).
Expect inference to drop from ~30–60s on CPU to ~5–10s on GPU.

### First deploy

The first deploy will be slow (~10–15 min) because Railway has to build
the ~8 GB image and TRIBE will lazy-load multi-GB of weights on the first
request. After deploy, hit `/warmup` once to trigger the load before any
real traffic.

```bash
curl -H "Authorization: Bearer $SERVICE_AUTH_TOKEN" \
     https://your-brain-service.up.railway.app/warmup
```

### Optional: pre-prime the full pipeline at startup

By default the service loads only the **TRIBE checkpoint** + **nilearn atlases** at startup. The first real `/render` call still pays a one-time download for WhisperX (~4 GB), Llama-3.2-3B (~6 GB), and Wav2Vec2 (~1 GB) — usually 5–10 min.

To eliminate that first-render delay, drop a short audio file at `brain-service/warmup-audio.wav` before building. The lifespan handler will run a full render on it during startup, populating every download cache.

Quick way to make one from any existing webm/mp3:

```powershell
ffmpeg -i path/to/any-recording.webm -t 3 -ar 16000 -ac 1 brain-service/warmup-audio.wav
```

3 seconds of any English speech is enough. The file ships inside the Docker image (~50–100 KB). Container startup will then take longer (~5–10 min for first deploy with cold caches; ~30s on subsequent deploys once caches land on the mounted volume), but the first real booth request is fast.

If you skip this, everything still works — the first attendee just waits longer.

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
