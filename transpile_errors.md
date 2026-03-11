# Transpilation Error Log

## Summary
- Total Stan models: 120
- Already transpiled: 34
- Remaining: 86

## Error Patterns
<!-- Track recurring errors here to avoid repeating mistakes -->

## Per-Model Errors
<!-- Logged automatically during transpilation -->

### eight_schools_centered (2026-03-11 07:59)
```
Exception: No API key. Set ANTHROPIC_API_KEY or pass api_key=
Traceback (most recent call last):
  File "/home/user/daemon/projects/posteriordb/run_transpile.py", line 96, in <module>
    results[model_name] = transpile_model(model_name)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/posteriordb/run_transpile.py", line 65, in transpile_model
    result = transpile_stan_to_pymc(
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/pymc_rust_compiler/stan_to_pymc.py", line 279, in transpile_stan_to_pymc
    raise ValueError("No API key. Set ANTHROPIC_API_KEY or pass api_key=")
ValueError: No API key. Set ANTHROPIC_API_KEY or pass api_key=

```

### eight_schools_centered (2026-03-11 08:03)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### logearn_interaction (2026-03-11 08:09)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```
