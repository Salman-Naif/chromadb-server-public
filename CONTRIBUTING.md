# Contributing to ChromaDB Knowledge Server

First off, thank you for considering contributing to this project! 🎉

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs what actually happened
- **Screenshots** if applicable
- **Environment details** (OS, Python version, etc.)

### 💡 Suggesting Features

Feature suggestions are welcome! Please:

- Use a **clear title** for the issue
- Provide a **detailed description** of the proposed feature
- Explain **why** this feature would be useful
- Include **examples** of how it would work

### 🔧 Pull Requests

1. **Fork** the repository
2. **Clone** your fork locally
3. Create a **new branch** for your feature/fix:
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. Make your changes
5. **Test** your changes thoroughly
6. **Commit** with clear messages:
   ```bash
   git commit -m "Add: amazing new feature"
   ```
7. **Push** to your branch:
   ```bash
   git push origin feature/amazing-feature
   ```
8. Open a **Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/Salman-Naif/chromadb-server-public.git
cd chromadb-server-public

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your settings

# Run development server
flask run --debug
```

## Code Style

- Follow **PEP 8** for Python code
- Use **meaningful variable names**
- Add **comments** for complex logic
- Keep functions **small and focused**

## Commit Message Format

Use clear, descriptive commit messages:

```
Add: new feature description
Fix: bug fix description
Update: changes to existing feature
Remove: removed feature/file
Docs: documentation changes
Style: formatting, no code change
Refactor: code restructuring
Test: adding tests
```

## Questions?

Feel free to open an issue with your question or reach out directly.

Thank you for contributing! 🙏
