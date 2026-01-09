import difflib
import hashlib
import subprocess
import sys
import time
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field


# --- Grammar ---
class OutputFormat(BaseModel):
    run_name: str = Field(
        ...,
        description="A kebab-case string (max 50 chars) combining the core logic change and key hyperparameter. E.g., 'layernorm-lr-decay'",
    )
    description: str = Field(
        ...,
        description="A concise technical summary (max 2 sentences) explaining the code refactor and config update.",
    )


# --- Automatic Experiment Scribe ---
class Autonym:
    def __init__(
        self,
        provider: Literal["ollama", "openai"] = "ollama",
        model_name: str = "phi3.5",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_diff_size: Optional[str] = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.max_diff_size = max_diff_size

        if self.provider == "openai":
            import openai

            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        else:
            import ollama

            self.client = ollama

    def _run_git_cmd(self, args: list) -> str:
        try:
            return (
                subprocess.check_output(args, stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError:
            return ""

    def _get_code_diff(self, base_ref: str, config_file: str) -> str:
        """Get code changes, EXCLUDING the config file itself."""
        cmd = ["git", "diff", base_ref, "--", ".", f":(exclude){config_file}"]
        return self._run_git_cmd(cmd)

    def _compute_config_diff(
        self, base_ref: str, config_file: str, runtime_config: Dict[str, Any]
    ) -> str:
        """
        Generates a synthetic diff between the committed config and the final in-memory dict.
        Normalizes both to string format to ignore comments/spacing changes.
        """
        # Fetch the BASELINE content from git
        try:
            baseline_content = self._run_git_cmd(
                ["git", "show", f"{base_ref}:{config_file}"]
            )
            if not baseline_content:
                return "New config file created."

            baseline_dict = yaml.safe_load(baseline_content)
        except Exception:
            # Fallback if file didn't exist in base_ref
            baseline_dict = {}

        # Normalize both to YAML strings
        baseline_str = yaml.dump(
            baseline_dict, sort_keys=True, default_flow_style=False
        )
        final_str = yaml.dump(runtime_config, sort_keys=True, default_flow_style=False)

        # Compute Diff
        diff_lines = difflib.unified_diff(
            baseline_str.splitlines(keepends=True),
            final_str.splitlines(keepends=True),
            fromfile=f"{config_file} (committed)",
            tofile=f"{config_file} (effective)",
            n=0,
        )

        return "".join(diff_lines)

    def _generate_unique_hash(self) -> str:
        timestamp = str(time.time()).encode("utf-8")
        return hashlib.sha256(timestamp).hexdigest()[:7]

    def _query_model(self, prompt: str) -> OutputFormat:
        try:
            if self.provider == "ollama":
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    format=OutputFormat.model_json_schema(),
                )
                return OutputFormat.model_validate_json(response["message"]["content"])

            elif self.provider == "openai":
                completion = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=OutputFormat,
                    temperature=0.1,
                )
                return completion.choices[0].message.parsed
        except Exception as e:
            print(f"Inference Error: {e}", file=sys.stderr)
            return None

    def summarize_run(
        self,
        runtime_config: Dict[str, Any],
        base_ref: str = "origin/master",
        config_path: str = "experiment.yaml",
        verbose: bool = False,
        dry_run: bool = False,
    ) -> Optional[OutputFormat]:

        print(f"Diffing against {base_ref}...", file=sys.stderr)

        code_diff = self._get_code_diff(base_ref, config_path)
        config_diff = self._compute_config_diff(base_ref, config_path, runtime_config)

        if verbose:
            print(
                f"\n=== Code Diff ({len(code_diff)} chars) ===",
                file=sys.stderr,
            )
            print(
                code_diff if code_diff.strip() else "(No logic changes detected)",
                file=sys.stderr,
            )
            print(
                f"\n=== Config Diff ({len(config_diff)} chars) ===",
                file=sys.stderr,
            )
            print(
                config_diff if config_diff.strip() else "(No config changes detected)",
                file=sys.stderr,
            )
            print(
                "====================================================\n",
                file=sys.stderr,
            )

        if not code_diff and not config_diff:
            print("No logic or config changes detected.", file=sys.stderr)
            meta = OutputFormat(run_name="baseline", description="Baseline run.")
            return meta
        prompt = f"""
        You are an MLOps assistant. Your goal is to generate metadata for a Weights & Biases (WandB) experiment based on code changes and configuration updates.

        I will provide two inputs:
        1. `GIT_DIFF`: The code changes. Focus ONLY on logic changes, model architecture modifications, and data pipeline adjustments. Ignore formatting or comments.
        2. `CONFIG_DIFF`: The experiment configuration changes. Focus on changed hyperparameters (learning rate, batch size, model size, etc.).

        ### INSTRUCTIONS:
        Generate a JSON object with exactly two keys:
        1. `run_name`: A short, slugified string (kebab-case) combining the most significant code change and the most important hyperparameter change. Max 50 chars.
        2. `description`: A concise, technical summary (max 2 sentences) explaining *what* changed in the code and *NOT* why.

        ### FORMAT:
        Return ONLY raw JSON. Do not use Markdown code blocks.
        ===CODE_CHANGES===
        {code_diff[:self.max_diff_size] if self.max_diff_size else code_diff} 

        ===EFFECTIVE_CONFIG_DIFF===
        {config_diff[:self.max_diff_size] if self.max_diff_size else config_diff}
        """

        if dry_run:
            print(f"\n=== [Autonym] Dry Run: Prompt Preview ===", file=sys.stderr)
            print(prompt, file=sys.stderr)
            print("=========================================\n", file=sys.stderr)
            return OutputFormat(
                run_name="dry-run", description="Dry run execution - no LLM called."
            )

        print(f"Querying {self.provider} ({self.model_name})...", file=sys.stderr)
        meta = self._query_model(prompt)
        if meta:
            unique_hash = self._generate_unique_hash()
            meta.run_name = f"{meta.run_name}-{unique_hash}"
            return meta
        return None
