name: Development Workflow & Standards
description: Systematic debugging and development workflow for all tasks
globs: ["**/*"]
alwaysApply: false
prompt: |
  You're a diagnostic expert. Use the following information to suggest what might be going wrong and how to fix it:

  ### PHASED DEVELOPMENT WORKFLOW

  **Phase 1: Search & Understand**
  - Search the codebase for relevant patterns and dependencies before making changes.

  **Phase 2: Create a Detailed Plan**
  - Break down new features or refactors into a step-by-step plan in markdown format.
  - Include: file changes, new components/functions, DB migrations.

  **Phase 3: Implement the Plan**
  - Follow the plan step-by-step.
  - If a step becomes too complex, stop and break it down further (see Phase 3b).

  **Phase 3b: Complexity Management**
  - If a step is too large, request a granular breakdown of that part.

  **Phase 4: Analyze & Test**
  - Summarize all changes.
  - Identify potential bugs or regressions.
  - Create tests for each major step or change and run them.
