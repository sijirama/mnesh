# Synthetic Data Prompt

Use the prompt below to generate session-level synthetic shell telemetry for Mnesh.

```text
You are generating realistic terminal session telemetry for training a next-command prediction model.

Output JSONL only.
Each line must be one command event from a multi-step shell session.
Do not explain anything.
Do not wrap in markdown.

Requirements:
- Generate complete sessions, not isolated commands.
- Each session must have 12 to 25 commands.
- Commands must be causally consistent and realistic.
- Sessions must include both monolithic workflows and mixed/transition workflows.
- Avoid overusing generic commands like ls, pwd, clear, cd, cat README.md.
- Generic commands may appear, but should not dominate sessions.
- Commands near the end of a session must not always determine the true workflow type.
- Include realistic transitions, such as:
  - python work followed by git staging
  - docker work followed by kubectl
  - frontend work followed by git commit
  - database debugging mixed with system inspection
- Preserve a coherent underlying session goal even when transitions happen.

For every command event, output these fields:
- session_id: string
- sequence_index: integer starting at 0
- cmd: exact shell command
- cmd_type: one of [filesystem, git, process, network, package, docker, k8s, python, node, system, text_processing, ssh, misc]
- session_context: one of [backend_dev, frontend_dev, devops, data_science, debugging, deployment, exploration, system_admin, open_source_contrib]
- git_enabled: boolean
- os: one of [linux, macos]
- shell: one of [zsh, bash, fish]
- project_ecosystem: one of [python, node, go, rust, infra, data, database, mixed]
- session_goal: short string label like "ship_api_fix", "train_model", "debug_prod_pod", "run_frontend_tests"
- primary_toolchain: short string like "django", "fastapi", "react", "nextjs", "docker", "kubernetes", "terraform", "postgres", "pandas", "go", "rust", "mixed"
- transition_state: one of [steady, setup, build, test, debug, deploy, stage_changes, handoff]
- repo_kind: one of [app, service, monorepo, infra, data_pipeline, notebook_repo, scripts, unknown]

Generation rules:
- Keep metadata consistent within a session unless the transition is intentional.
- If git_enabled is true, git commands should be plausible for the repo/workflow.
- If a session is primarily python, a late git add command should not erase the session’s python identity in metadata.
- Include some sessions where the final 1-2 commands are misleading relative to the main workflow.
- Include some sessions where the final 1-2 commands truly reflect the next likely action.
- Make command strings concrete and varied.
- Prefer plausible developer commands over toy examples.
- Include errors, retries, inspections, and follow-up commands when realistic.
- Do not generate duplicate sessions.

Dataset balance:
- Roughly balance across python, node/frontend, docker/k8s, git-heavy, sysadmin, database, go, rust, and data workflows.
- Include at least 30% mixed sessions with meaningful transitions.
- Limit trivial/generic commands to less than 15% of all rows.

Additional constraint:
- For at least 25% of sessions, make the last command belong to a different cmd_type than the dominant workflow, while keeping the session_goal and project_ecosystem tied to the dominant workflow.

Output JSONL only.
```

## Why This Prompt

- It keeps the data close to the current Mnesh session shape.
- It adds higher-signal metadata for future supervision:
  - `project_ecosystem`
  - `session_goal`
  - `primary_toolchain`
  - `transition_state`
  - `repo_kind`
- It directly targets the current failure mode: the model overfitting to the final command in mixed sessions.

## Example Row

```json
{"session_id":"sess_001_python_api","sequence_index":9,"cmd":"git add apps/accounts/models.py","cmd_type":"git","session_context":"backend_dev","git_enabled":true,"os":"linux","shell":"zsh","project_ecosystem":"python","session_goal":"ship_auth_fix","primary_toolchain":"django","transition_state":"stage_changes","repo_kind":"service"}
```

## Recommended Usage

1. Generate full sessions in JSONL.
2. Validate schema and metadata consistency.
3. Reject sessions with too many generic commands.
4. Reject sessions with contradictory metadata/command patterns.
5. Mix synthetic data with existing data rather than replacing the current dataset entirely.
