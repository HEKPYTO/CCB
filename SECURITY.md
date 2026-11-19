# Security

## Supported versions

Only the latest release receives fixes.

## Reporting a vulnerability

Open a GitHub security advisory on the repository. Do not file a public
issue for a vulnerability.

## Notes

The tool is local-first: it reads a local file and writes to the paths given on
the command line. It makes no network calls, collects no telemetry, and does
not execute input. Treat input files and output directories as untrusted: run
`ccb` without elevated privileges and write outputs to paths that do not
contain sensitive files.
