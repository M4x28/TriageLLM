"""Endpoint configuration for behavioral (Bloom) evaluation.

Single source of truth for how the behavioral pipeline reaches its models.
Everything is local: the target model and the auditor/judge model are each
served behind a vLLM OpenAI-compatible endpoint, and Bloom (on the inspect_ai
substrate) reaches them with service-prefixed model strings.

inspect_ai's `openai-api/<service>/<model>` provider derives two env vars from
the service prefix: `<SERVICE>_BASE_URL` and `<SERVICE>_API_KEY`. So each role
gets its own service prefix -> its own base URL, and the local-vs-API choice is
a one-line change here (point a service at an API base URL + real key instead of
a local vLLM port).

Spike-verified facts baked in below:
- vLLM must be served with `--enable-auto-tool-choice --tool-call-parser hermes`
  (Qwen tool calling) and a context window large enough for Bloom's up-to-8192
  completion tokens (we use 32768).
- Roles used by `bloom scenarios` + `inspect eval petri_bloom/bloom_audit`:
  scenarios, auditor, target, judge.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Endpoint:
    """One served model behind an OpenAI-compatible base URL.

    `service` is the inspect_ai service prefix; `model` is the served-model-name
    exposed by the vLLM server (must match `--served-model-name`).
    """
    service: str          # inspect_ai service prefix (-> <SERVICE>_BASE_URL/_API_KEY)
    model: str            # served-model-name on the vLLM server
    port: int             # local vLLM port
    host: str = "localhost"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def model_string(self) -> str:
        """inspect_ai model string, e.g. 'openai-api/target/qwen3-1.7b'."""
        return f"openai-api/{self.service}/{self.model}"

    def export_env(self) -> None:
        """Set the env vars inspect_ai reads for this service prefix."""
        prefix = self.service.upper().replace("-", "_")
        os.environ[f"{prefix}_BASE_URL"] = self.base_url
        # vLLM ignores the key, but inspect_ai requires it to be present.
        os.environ.setdefault(f"{prefix}_API_KEY", "local-dummy-key")


# vLLM serving flags required for Bloom (see serve_model.sh). Kept here so the
# requirements are documented next to the endpoint definitions.
VLLM_SERVE_FLAGS = (
    "--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768"
)

# Default local topology. The target is our model under evaluation; the auditor
# and judge share one strong local model (one server, two roles). Swap a service
# to an external API later by changing base_url/key via env without touching
# callers.
DEFAULT_TARGET = Endpoint(service="target", model="qwen3-1.7b", port=8001)
DEFAULT_AUDITOR = Endpoint(service="auditor", model="qwen3-8b", port=8002)
# Judge reuses the auditor server/model by default (same strong local model).
DEFAULT_JUDGE = Endpoint(service="judge", model="qwen3-8b", port=8002)


def configure_local(target: Endpoint = DEFAULT_TARGET,
                    auditor: Endpoint = DEFAULT_AUDITOR,
                    judge: Endpoint = DEFAULT_JUDGE) -> dict[str, str]:
    """Export env for all roles and return the inspect_ai model-role strings.

    `scenarios` (used by `bloom scenarios`) reuses the auditor model.
    """
    for ep in {target, auditor, judge}:  # set-dedupes shared host:port services
        ep.export_env()
    return {
        "scenarios": auditor.model_string,
        "auditor": auditor.model_string,
        "target": target.model_string,
        "judge": judge.model_string,
    }
