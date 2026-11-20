# Portable compact archive audit

This CPU-only verifier checks a private PR54 compact archive without unpacking it. It requires Python and `zstandard`; it does not import a model runtime, load tensors, use a GPU, delete evidence or access remote hosts.

From the DFlash-Online directory, run tests with:

```bash
PYTHONPATH=. python -B -m unittest experiments.fresh_mathqa_endpoints.archive_audit.test_verify_archive -v
```

Invoke `python -m experiments.fresh_mathqa_endpoints.archive_audit.verify_archive --help` for the private archive, member-manifest, transport, keeper-stat and authentication receipt arguments. Every path is supplied explicitly. Choose a new output directory so earlier failed audits remain intact.

The original remote delivery binds the member manifest. A recompression receipt separately binds the received archive's size and SHA256; complete member verification ties both together. A passed `CLOSEOUT.json` requires both full stream verification and keeper/retirement reconciliation. Synthetic tests do not establish real delivery. The audit outputs contain private paths from their inputs and require sanitization before publication.
