# Docker Image Build Policy

- **Production images** must be built and published only by CI/CD (GitHub Actions) on official releases. Local building of production images is strictly prohibited to ensure proper security audits and compliance.
- **Local builds** are allowed only for test images (e.g., `supervisely/prompt-based-image-filtering:test`).
- If you need to use a custom SDK branch or commit, specify it in `dev_requirements.txt`.
