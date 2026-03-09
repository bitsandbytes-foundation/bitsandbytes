# Security Policy

## Supported Versions

We provide security updates for the latest stable minor release line.

| Version  | Supported |
| -------- | --------- |
| 0.49.x   | ✅        |
| < 0.49.x | ❌        |

> Note: Pre-releases, development builds, and commits on `main` are not considered supported release versions. If you believe you have found a vulnerability in unreleased code, please still report it following the process below.

## Reporting a Vulnerability

Please report security issues **privately** using the GitHub Security Advisory tool to create a new draft advisory:

- https://github.com/bitsandbytes-foundation/bitsandbytes/security/advisories/new

Do not open a public GitHub issue for security-sensitive reports.

### What to include

To help us triage and respond quickly, please include:

- A clear description of the issue and potential impact
- Affected version(s) and environment details (OS, GPU type, CUDA version, Python version, PyTorch version, etc)
- Steps to reproduce (ideally a minimal proof of concept)
- Any relevant logs, crash traces, or screenshots
- Any known mitigations or workarounds

## Response process

We will review reports filed via GitHub Security Advisories and collaborate with the reporter in the advisory thread to:

- Confirm and reproduce the report
- Assess severity and affected versions
- Identify mitigations and/or prepare a fix
- Coordinate any follow-up needed prior to broader communication
