# Project Context & Guidelines

## Git Operations
- When executing git pushes to remote, the CLI shell environment passes an empty SSH command by default. Always execute git push with:
  `git -c core.sshcommand=ssh push <remote> <branch>`
