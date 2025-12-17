# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **Do NOT** open a public issue
2. Email the details to: [snma2003@outlook.sa]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Security Best Practices

When deploying this server:

1. **Change default credentials** - Never use default admin password in production
2. **Use HTTPS** - Always deploy behind HTTPS (Render.com provides this automatically)
3. **Set strong SECRET_KEY** - Generate a random key for production
4. **Limit access** - Use firewall rules to restrict access if needed
5. **Regular updates** - Keep dependencies updated
6. **Monitor logs** - Check error logs for suspicious activity

## Environment Variables

Never commit these to version control:
- `SECRET_KEY`
- `OPENROUTER_API_KEY`
- `GEMINI_API_KEY`
- `ADMIN_PASSWORD`
- `DATABASE_URL`

Use `.env` file locally and environment variables in production.
