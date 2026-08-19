# TLS certificates for the Nginx reverse proxy

`nginx.conf` expects two files here:
- `ssense.crt`
- `ssense.key`

This directory is git-ignored (see `.gitignore` — `*.key`, `*.crt` patterns)
so real certificates never end up committed.

## Local development (self-signed)

```bash
openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout ssense.key \
  -out ssense.crt \
  -subj "/CN=localhost"
```

Browsers/extensions will show a certificate warning against a self-signed
cert — expected for local dev, not something to "fix" here.

## Production

Use a real certificate authority. Two common paths:

**Let's Encrypt / certbot** (if this box has a public domain pointed at it):
```bash
certbot certonly --standalone -d yourdomain.example.com
# then symlink or copy the issued files to ssense.crt / ssense.key
```

**Internal/enterprise CA** — request a cert for the server's real hostname
from your org's CA and place the issued cert + key here under the same
filenames.

Whichever path is used, set up **automatic renewal** before this goes live
long-term — Let's Encrypt certs expire every 90 days, and an expired cert
here means the entire proxy stops serving TLS.
