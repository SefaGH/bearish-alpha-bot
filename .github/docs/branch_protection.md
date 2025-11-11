# Branch Protection Policy

## Purpose

This document defines the branch protection rules for the Bearish Alpha Bot repository to ensure code quality, stability, and proper review processes.

## Protected Branches

### `main` Branch

The `main` branch is the primary protected branch containing production-ready code.

#### Protection Rules

1. **Require Pull Request Reviews**
   - Minimum reviewers: **1**
   - Dismiss stale pull request approvals when new commits are pushed: **Yes**
   - Require review from Code Owners: **Optional** (when CODEOWNERS file is present)

2. **Require Status Checks**
   - Require branches to be up to date before merging: **Yes**
   - Required status checks that must pass:
     - ✅ **Core Tests** (`tests.yml` / `test-with-docker` job)
     - ✅ **Python Version Check** - Validates Python 3.11 usage
     - ⚠️ **Static Analysis** (when re-enabled: `static_check.yml`)
     - ⚠️ **Integration Tests** (when re-enabled: `integration-tests.yml`)

3. **Require Linear History**
   - Enforce linear history: **Yes**
   - This prevents merge commits and keeps history clean
   - Use "Squash and merge" or "Rebase and merge" strategies

4. **Restrict Direct Pushes**
   - Allow force pushes: **No**
   - Allow deletions: **No**
   - Restrict who can push to matching branches:
     - Repository administrators only
     - No direct commits to `main`
     - All changes must go through Pull Requests

5. **Additional Protections**
   - Require signed commits: **Optional** (recommended for security)
   - Include administrators: **Yes** (administrators also follow the rules)
   - Require conversation resolution before merging: **Yes**

## Status Checks

### Core Tests (`tests.yml`)

**Purpose:** Validates that all core functionality works correctly

**Requirements:**
- Docker build succeeds
- Python 3.11 environment verified
- All unit tests pass
- Dependencies install correctly
- aiohttp 3.8.6 compatibility verified

**When it runs:**
- On every push to PR
- On every commit to protected branches

**Failure handling:**
- PR cannot be merged if this check fails
- Review logs and fix issues before merging

### Static Analysis (`static_check.yml`)

**Purpose:** Ensures code quality and security standards

**Tools:**
- **ruff**: Code style and linting
- **mypy**: Type checking
- **bandit**: Security vulnerability scanning
- **pyright**: Additional type checking

**Requirements:**
- No style violations
- No type errors
- No security vulnerabilities
- All code meets PEP 8 standards

**Status:** Currently disabled, will be re-enabled after Phase-2 housekeeping

### Integration Tests (`integration-tests.yml`)

**Purpose:** Validates multi-component interactions

**Coverage:**
- API integrations (BingX, Binance, KuCoin)
- WebSocket connections
- Database operations
- End-to-end workflows

**Status:** Currently disabled, needs assessment for Phase-2

### Python Version Check (`python-version-check.yml`)

**Purpose:** Ensures Python 3.11 is used consistently

**Checks:**
- All workflow files specify Python 3.11
- No Python 3.12+ references
- `.python-version` file contains 3.11
- `pyproject.toml` specifies correct Python version

**Criticality:** HIGH - Python version mismatches cause build failures

## Pull Request Process

### Before Creating a PR

1. **Branch from `main`**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature
   ```

2. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat: description of changes"
   ```

3. **Run local checks**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Run linter
   ruff check src/
   
   # Run type checker
   mypy src/
   
   # Verify Python version
   python --version  # Must be 3.11.x
   ```

4. **Push to GitHub**
   ```bash
   git push origin feature/your-feature
   ```

### Creating the PR

1. Go to GitHub and create Pull Request
2. Fill in the PR template:
   - Title: Clear, descriptive title
   - Description: What changed and why
   - Checklist: Complete all items
   - Related issues: Link relevant issues

3. Add appropriate labels:
   - `bug` - Bug fixes
   - `feature` - New features
   - `documentation` - Documentation updates
   - `housekeeping` - Code cleanup
   - `ci` - CI/CD changes
   - `phase:2` / `phase:3` - Phase-specific work

### PR Review Process

1. **Automated Checks**
   - CI builds and runs tests
   - Status checks must pass
   - Python version validated

2. **Code Review**
   - At least 1 reviewer required
   - Reviewers check:
     - Code quality and style
     - Test coverage
     - Documentation updates
     - Security considerations
     - Python 3.11 compatibility

3. **Address Feedback**
   - Respond to comments
   - Make requested changes
   - Push updates (triggers new checks)

4. **Approval and Merge**
   - Once approved and checks pass
   - Use "Squash and merge" (preferred) or "Rebase and merge"
   - Delete branch after merge

### PR Checklist

Every PR should meet these criteria:

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Python 3.11 verified
- [ ] No merge conflicts
- [ ] Commit messages follow conventions
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Breaking changes documented

## Merge Strategies

### Preferred: Squash and Merge

**When to use:** Most feature branches and bug fixes

**Advantages:**
- Clean, linear history
- Single commit per feature
- Easy to revert
- Clear commit messages

**How:**
1. GitHub combines all commits into one
2. Edit the combined commit message
3. Merge to `main`

### Alternative: Rebase and Merge

**When to use:** When individual commits are well-crafted and meaningful

**Advantages:**
- Preserves individual commits
- Maintains detailed history
- Good for complex features

**How:**
1. Rebase feature branch onto `main`
2. Resolve any conflicts
3. Merge to `main`

### Never: Merge Commit

**Reason:** Creates non-linear history, harder to maintain

**Exception:** Only for merging release branches or major version updates

## Emergency Procedures

### Hotfix Process

For critical production issues:

1. **Create hotfix branch from `main`**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b hotfix/critical-issue
   ```

2. **Make minimal fix**
   - Fix only the critical issue
   - Add regression test
   - Update version if needed

3. **Fast-track review**
   - Label as `priority: critical`
   - Request immediate review
   - Notify team in appropriate channels

4. **Deploy immediately after merge**

### Bypassing Protections

**When allowed:**
- Never under normal circumstances
- Only repository administrators
- Only for critical emergencies
- Requires documentation of why

**Process:**
1. Document the reason
2. Make the change
3. Create follow-up PR for proper review
4. Post-mortem to prevent future need

## Enforcement

### Who Enforces

- GitHub: Automated enforcement via branch protection settings
- Reviewers: Manual enforcement during code review
- CI/CD: Automated testing and validation
- Repository Administrators: Policy updates and exceptions

### Violations

**Automated Violations:** Prevented by GitHub protections

**Manual Violations:** Caught during review
- PR rejected
- Changes requested
- Education on proper process

### Updates to Policy

This policy is reviewed and updated:
- Quarterly
- After major project phases
- When issues are identified
- As project needs evolve

## Quick Reference

### ✅ Do's

- Create PRs for all changes
- Write clear commit messages
- Run tests before pushing
- Respond to review feedback
- Keep PRs focused and small
- Use Python 3.11
- Update documentation

### ❌ Don'ts

- Don't push directly to `main`
- Don't merge without approval
- Don't ignore failing checks
- Don't use Python 3.12+
- Don't skip tests
- Don't leave unresolved conversations
- Don't force push to PRs under review

## Status Check Details

### Required Checks

| Check | Type | Criticality | Timeout |
|-------|------|-------------|---------|
| Core Tests | Automated | HIGH | 10 min |
| Python Version | Automated | HIGH | 2 min |
| Static Analysis | Automated | MEDIUM | 5 min |
| Integration Tests | Automated | MEDIUM | 15 min |
| Code Review | Manual | HIGH | N/A |

### Optional Checks

| Check | Type | Purpose |
|-------|------|---------|
| Coverage Report | Automated | Track test coverage |
| Performance Tests | Automated | Detect regressions |
| Security Scan | Automated | Vulnerability detection |
| Docs Build | Automated | Verify docs build |

## Contact

For questions about branch protection policy:
- Open an issue with label `question`
- Discussion in GitHub Discussions
- Contact repository administrators

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-11  
**Next Review:** 2026-02-11  
**Owner:** Repository Administrators
