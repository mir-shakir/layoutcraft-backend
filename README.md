# LayoutCraft Backend

The core orchestration engine for **LayoutCraft**, a SaaS platform for generating high-fidelity, dimensionally accurate marketing assets using AI. This service bridges the gap between generative models and production-ready design by enforcing strict layout constraints that diffusion models cannot achieve on their own.

**Live SaaS:** [https://layoutcraft.tech](https://layoutcraft.tech)

---

## The Engineering Problem

Standard generative image models (e.g., Midjourney, Stable Diffusion) operate on pixel probability. While creative, they fundamentally lack:

* **Dimensional Accuracy**
  You cannot ask a diffusion model to *"make this banner exactly 1200x630px with 40px padding."*

* **Text Fidelity**
  Raster generation often results in hallucinated or unreadable glyphs.

* **Iterative Editing**
  Changing one element (e.g., *"move the logo left"*) typically regenerates the entire image, losing the original context.

---

## The Solution

LayoutCraft solves this by treating design as a **Semantic Specification** rather than a pixel array.

The backend uses LLMs to generate a structural blueprint (JSON / HTML / CSS), which is then deterministically rendered by a headless browser engine. This ensures:

* 100% text clarity
* Pixel-perfect dimensions

---

## Architecture

The backend is built as a high-concurrency microservice using **FastAPI**.

---

## Tech Stack

* **Framework:** Python (FastAPI)
* **AI Orchestration:** Google Vertex AI (Primary) + Gemini API (Fallback)
* **Database & Auth:** Supabase (PostgreSQL + GoTrue)
* **Rendering:** Playwright (Headless Chromium)
* **Payments:** Dodo Payments Integration
* **Deployment:** Render (PaaS) with Docker

---

## Core Workflows

### 1. Intent Parsing

User prompts are analyzed to extract design intent
(e.g., *"Instagram Story for a summer sale"*).

### 2. Semantic Generation

* The system constructs a prompt context and queries Vertex AI.
* The LLM returns a structured JSON payload defining:

  * DOM structure
  * Computed styles
  * Asset placement
* **Resilience:** Retry logic with exponential backoff handles AI provider rate limits and latency.

### 3. Algorithmic Rendering

* The JSON specification is hydrated into a sandboxed HTML/CSS environment.
* Playwright captures the viewport at exact dimensions defined by the user
  (e.g., OG tags, Instagram Stories).

### 4. State Management

* Every generation is versioned.
* Users can iteratively refine designs.
* The backend calculates the **semantic diff** between the current state and the user’s edit request
  (e.g., *"Make the background blue"*), modifying only the relevant DOM nodes without breaking layout.

---

## Technical Highlights

### Concurrency & Error Handling

Rendering high-fidelity assets is CPU-intensive. The application utilizes FastAPI’s `async/await` capabilities to handle concurrent generation requests without blocking the main event loop.

* **Retries:** Custom decorators manage API timeouts and transient failures from upstream AI providers.
* **Resource Management:** Headless browser contexts are managed with strict lifecycles to prevent memory leaks during high-load periods.

---

## SaaS Integration

This repository includes complete webhook handling logic for **Dodo Payments**, managing the subscription lifecycle:

* Subscription upgrades and downgrades
* Usage quota tracking (Free vs. Pro tiers) stored in Supabase
* Secure webhook signature verification

---

## Development Constraints

**Note:** Running this project locally requires access to restricted production infrastructure.

* **Vertex AI Credentials**
  Requires an authenticated Google Cloud Service Account with specific quota allocations.

* **Supabase Project**
  Requires connection to the production or staging database schema.

* **Environment Variables**
  A `.env` file with API keys for Dodo Payments, Google Cloud, and Supabase is required.

For security reasons, these credentials are not included in the repository. Please refer to the codebase to understand the architectural patterns and implementation details.
