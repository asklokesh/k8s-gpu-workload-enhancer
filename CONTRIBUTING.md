# Contributing to KGWE

Thank you for your interest in contributing to the Kubernetes GPU Workload Enhancer (KGWE)! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful, inclusive, and considerate in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Go 1.21 or later
- Python 3.10 or later
- Docker 24.x or later
- Kubernetes 1.26+ cluster (for testing)
- Helm 3.x
- Access to NVIDIA GPUs (for full testing)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/k8s-gpu-workload-enhancer.git
   cd k8s-gpu-workload-enhancer
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/asklokesh/k8s-gpu-workload-enhancer.git
   ```

## Development Setup

### Install Dependencies

```bash
# Go dependencies
go mod download

# Python dependencies
cd src/optimizer
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install development tools
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
pip install ruff mypy pytest pytest-cov
```

### Build the Project

```bash
# Build all Go components
make build

# Run linters
make lint

# Run tests
make test
```

### Local Development

For local development without a GPU cluster:

```bash
# Run the controller locally (uses ~/.kube/config)
make run-controller

# Run the optimizer service
make run-optimizer
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

| Type | Description |
|------|-------------|
| Bug fixes | Fix issues reported in GitHub Issues |
| Features | Implement new functionality |
| Documentation | Improve or add documentation |
| Tests | Add or improve test coverage |
| Performance | Optimize existing code |
| Refactoring | Improve code quality without changing behavior |

### Finding Issues to Work On

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` are ready for community contribution
- Check the project roadmap in the PRD for planned features

### Reporting Bugs

When reporting bugs, please include:

1. **Environment details**: OS, Go version, Python version, K8s version
2. **GPU details**: GPU model, driver version, CUDA version
3. **Steps to reproduce**: Minimal steps to reproduce the issue
4. **Expected behavior**: What you expected to happen
5. **Actual behavior**: What actually happened
6. **Logs**: Relevant log output (sanitized of sensitive data)

### Suggesting Features

Feature requests should include:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives considered**: Other approaches you've thought about
4. **Additional context**: Mockups, examples, or references

## Pull Request Process

### Before Submitting

1. **Create an issue first** for significant changes
2. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### PR Requirements

All pull requests must:

- [ ] Pass all CI checks (lint, build, test)
- [ ] Include tests for new functionality
- [ ] Update documentation as needed
- [ ] Follow the coding standards
- [ ] Have a clear, descriptive title
- [ ] Reference related issues

### PR Title Format

Use conventional commit format:

```
type(scope): description

Examples:
feat(scheduler): add NVSwitch topology detection
fix(mig): resolve partition cleanup race condition
docs(readme): update installation instructions
test(optimizer): add workload classification tests
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

### Review Process

1. Submit your PR
2. Entelligence.ai will provide automated code review
3. Maintainers will review within 3-5 business days
4. Address feedback and update your PR
5. Once approved, a maintainer will merge

## Coding Standards

### Go Code

Follow the [Effective Go](https://go.dev/doc/effective_go) guidelines.

```go
// Good: Clear, idiomatic Go
func (s *Scheduler) ScoreNode(node *NodeTopology, req GPURequirements) (int, error) {
    if node == nil {
        return 0, errors.New("node cannot be nil")
    }
    // Implementation...
}

// Avoid: Unclear naming, missing error handling
func (s *Scheduler) sn(n *NodeTopology, r GPURequirements) int {
    // Implementation without error handling...
}
```

**Go Style Guidelines:**
- Use `gofmt` for formatting
- Run `golangci-lint` before committing
- Keep functions focused and under 50 lines
- Use meaningful variable names
- Add comments for exported functions

### Python Code

Follow [PEP 8](https://peps.python.org/pep-0008/) and use type hints.

```python
# Good: Type hints, docstrings, clear structure
def predict_resources(
    self,
    workload_id: str,
    model_params_billions: float | None = None,
) -> ResourcePrediction:
    """
    Predict optimal resources for a workload.

    Args:
        workload_id: Unique workload identifier
        model_params_billions: Model size in billions of parameters

    Returns:
        ResourcePrediction with recommended configuration
    """
    # Implementation...
```

**Python Style Guidelines:**
- Use `ruff` for linting and formatting
- Use `mypy` for type checking
- Write docstrings for all public functions
- Keep functions under 30 lines when possible

### Kubernetes Manifests

```yaml
# Good: Clear labels, resource limits, comments
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kgwe-scheduler
  labels:
    app.kubernetes.io/name: kgwe
    app.kubernetes.io/component: scheduler
spec:
  replicas: 2
  selector:
    matchLabels:
      app.kubernetes.io/name: kgwe
      app.kubernetes.io/component: scheduler
  template:
    spec:
      containers:
        - name: scheduler
          resources:
            requests:
              cpu: 200m
              memory: 512Mi
            limits:
              cpu: 1000m
              memory: 1Gi
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests (no external dependencies)
├── integration/    # Integration tests (require K8s)
└── e2e/           # End-to-end tests (require GPU cluster)
```

### Writing Tests

**Go Tests:**
```go
func TestScheduler_ScoreNode(t *testing.T) {
    tests := []struct {
        name     string
        node     *NodeTopology
        req      GPURequirements
        wantErr  bool
        minScore int
    }{
        {
            name: "optimal NVLink placement",
            node: createTestNode(8, true), // 8 GPUs with NVLink
            req:  GPURequirements{GPUCount: 4, TopologyPreference: TopologyPreferenceNVLinkOptimal},
            minScore: 80,
        },
        // More test cases...
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            s := NewScheduler(DefaultConfig())
            score, err := s.ScoreNode(tt.node, tt.req)

            if (err != nil) != tt.wantErr {
                t.Errorf("unexpected error: %v", err)
            }
            if score < tt.minScore {
                t.Errorf("score %d < minimum %d", score, tt.minScore)
            }
        })
    }
}
```

**Python Tests:**
```python
import pytest
from workload_optimizer import WorkloadOptimizer, ResourcePrediction

class TestWorkloadOptimizer:
    @pytest.fixture
    def optimizer(self):
        return WorkloadOptimizer()

    def test_predict_resources_small_model(self, optimizer):
        """Test resource prediction for small models."""
        prediction = optimizer.predict_resources(
            workload_id="test-001",
            hints={"model_params_billions": 0.5}
        )

        assert prediction.recommended_gpus == 1
        assert prediction.recommended_memory_gb >= 10
        assert 0 < prediction.confidence <= 1

    @pytest.mark.parametrize("model_size,expected_gpus", [
        (7, 1),
        (13, 2),
        (70, 8),
    ])
    def test_gpu_scaling_with_model_size(self, optimizer, model_size, expected_gpus):
        """Test that GPU count scales appropriately with model size."""
        prediction = optimizer.predict_resources(
            workload_id=f"test-{model_size}b",
            hints={"model_params_billions": model_size}
        )
        assert prediction.recommended_gpus == expected_gpus
```

### Running Tests

```bash
# Run all tests
make test

# Run Go tests with coverage
go test -v -race -coverprofile=coverage.out ./...

# Run Python tests
cd src/optimizer && pytest -v --cov=.

# Run integration tests (requires K8s cluster)
make integration-test

# Run E2E tests (requires GPU cluster)
make e2e-test
```

## Documentation

### Where to Document

| Content Type | Location |
|--------------|----------|
| API Reference | Code comments + `docs/api-reference.md` |
| User Guides | `docs/` directory |
| Architecture | `docs/architecture.md` |
| Examples | `examples/` directory |
| Inline Comments | Source code |

### Documentation Standards

- Use clear, concise language
- Include code examples where helpful
- Keep documentation up-to-date with code changes
- Add diagrams for complex concepts (use Mermaid or ASCII art)

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions

### Getting Help

If you need help:

1. Check existing documentation
2. Search closed issues for similar problems
3. Open a new issue with the `question` label

### Recognition

Contributors are recognized in:

- Git commit history
- Release notes for significant contributions
- CONTRIBUTORS.md file (for regular contributors)

---

## License

By contributing to KGWE, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to KGWE! Your efforts help make GPU workload management better for everyone.
