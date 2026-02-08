# Copilot instructions for LayoutCraft backend

## Big picture
- FastAPI service that turns prompts into HTML/CSS then renders with Playwright for pixel-accurate assets; core app + endpoints live in [index.py](../index.py).
- Prompt engineering is centralized in [prompts/design_prompts.py](../prompts/design_prompts.py); edits use `HTML_EDIT_PROMPT` with minimal diffs.
- Dimension presets and prompt context helpers live in [config/dimension_presets.py](../config/dimension_presets.py); use preset names like `blog_header`, `story`, `social_square`.

## Architecture & boundaries
- HTTP routes are split by domain under [routes/](../routes/) (auth, users, payments, advanced generation). Keep route handlers thin.
- Business logic and Supabase access live in [services/](../services/) (examples: [services/generation_service.py](../services/generation_service.py), [services/template_service.py](../services/template_service.py)).
- Pydantic request/response models live in [models/](../models/) (e.g., [models/generation.py](../models/generation.py)).
- Authentication uses local JWT validation against Supabase in [auth/middleware.py](../auth/middleware.py); route deps are in [auth/dependencies.py](../auth/dependencies.py).

## Project-specific patterns
- Use `get_auth_middleware()` to access the shared Supabase client; don’t instantiate your own client in routes.
- Tier gating is enforced via dependencies like `require_pro_plan` / `RequireProTier` and tier config in [config/tier_config.py](../config/tier_config.py) and [services/premium_service.py](../services/premium_service.py).
- Generation history uses `design_thread_id` + `parent_id` to group edits; keep these fields consistent with [models/generation.py](../models/generation.py).
- Retry behavior for intermittent SSL errors is handled via `retry_on_ssLError` in [config/decorators.py](../config/decorators.py).

## Integrations & external deps
- Supabase tables accessed in code: `user_profiles`, `generations`, `generation_history`, `brand_kits`, `custom_templates`.
- AI providers: Google Gemini via `google-genai` (see initialization in [index.py](../index.py)).
- Rendering: Playwright headless Chromium; exports use Pillow + CairoSVG in [services/export_service.py](../services/export_service.py).
- Payments/webhooks: see [routes/dodo.py](../routes/dodo.py) and [routes/paddle.py](../routes/paddle.py).

## Local dev constraints
- Running locally requires secrets for Supabase + Gemini (see env usage in [auth/middleware.py](../auth/middleware.py) and [index.py](../index.py)).
- No test or run commands are documented in-repo; verify any new workflow before adding instructions.
