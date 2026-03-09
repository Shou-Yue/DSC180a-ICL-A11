# RQ3 Test Suite

Test utilities and validation scripts for the RQ3 commercial LLM framework.

## Available Tests

### API Connection Tests
**File:** `test_api_connections.py`

Tests connectivity to all three LLM providers (Gemini, Claude, GPT-4o-mini).
Verifies that API keys are configured and models are accessible.

```bash
cd ../..  # Navigate to src/icl_reproduction
python -m rq3.tests.test_api_connections
```

### Claude Model Discovery
**File:** `find_claude_model.py`

Diagnostic tool to discover which Claude models are available on your account.
Useful when API model names change or become unavailable.

```bash
python -m rq3.tests.find_claude_model
```

### Phase 2: Dataset Generation Tests
**File:** `test_phase2.py`

Validates the BinaryClassificationDataset wrapper:
- Dataset instantiation and shape verification
- Tensor dimension correctness
- Prompt formatting functionality
- Label flipping probability

```bash
python tests/test_phase2.py
```

### Phase 3: TinyLlama Probing Tests
**File:** `test_phase3.py`

Framework validation for open-weights model probing:
- Module imports and available functions
- Gradient descent baseline computation
- Preconditioned GD variant
- Prompt formatting for linear regression
- Function signature verification

```bash
python tests/test_phase3.py
```

## Quick Test Run

Run all tests from the RQ3 directory:

```bash
cd ..
python -c "from tests.test_phase2 import *; exec(open('tests/test_phase2.py').read())"
python -c "from tests.test_phase3 import *; exec(open('tests/test_phase3.py').read())"
```

## Requirements

All tests require:
- `.env` file with API keys (Gemini, Claude, GPT)
- Python 3.11+
- PyTorch
- google-genai, anthropic, openai packages

See parent directory's `requirements.txt` for full dependencies.
