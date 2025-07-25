name: Implement Feature From Spec
description: Follows a detailed spec to create or edit code in the project
globs: ["**/*"]
alwaysApply: true
prompt: |
  You will implement the following feature plan into the current project.

  The plan may include:
  - Code architecture
  - Threading or concurrency design
  - Library recommendations
  - Setup instructions

  ### YOUR JOB

  1. Parse and understand the plan
  2. Break it into specific dev tasks (1 task = 1 commit)
  3. Identify the files and line ranges to be modified
  4. Execute each step:
     - Add/modify code
     - Log changes
     - Keep code idiomatic and readable

  ### CONTEXT

  Feature Plan:
  ---
  {{selection}}
  ---

  ### RULES

  - Do NOT hallucinate functionality not mentioned in the plan.
  - Always preserve existing logic unless a replacement is specified.
  - If a function is new, add docstrings and TODO markers.
  - Do not modify unrelated files or logic.

  Respond only with what you're doing or changing — no extra explanation.

agent: true
