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

### radon_county (2026-03-11 08:16)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### radon_hierarchical_intercept_noncentered (2026-03-11 08:22)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### radon_partially_pooled_centered (2026-03-11 08:24)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### GLM_Poisson_model (2026-03-11 08:30)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### pilots (2026-03-11 08:31)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### logmesquite_logvas (2026-03-11 08:32)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### GLMM1_model (2026-03-11 08:35)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### radon_variable_intercept_slope_noncentered (2026-03-11 08:36)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### seeds_model (2026-03-11 08:43)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### Mtbh_model (2026-03-11 08:43)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### dogs_log (2026-03-11 08:47)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### seeds_centered_model (2026-03-11 08:48)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### garch11 (2026-03-11 08:49)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### logistic_regression_rhs (2026-03-11 08:50)
```
Exception: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 207286 tokens > 200000 maximum'}, 'request_id': 'req_011CYw1JZbKcWtcfR9qP1qGA'}
Traceback (most recent call last):
  File "/home/user/daemon/projects/posteriordb/run_transpile.py", line 96, in <module>
    results[model_name] = transpile_model(model_name)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/posteriordb/run_transpile.py", line 65, in transpile_model
    result = transpile_stan_to_pymc(
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/pymc_rust_compiler/stan_to_pymc.py", line 341, in transpile_stan_to_pymc
    response = client.messages.create(
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/_utils/_utils.py", line 282, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/resources/messages/messages.py", line 996, in create
    return self._post(
           ^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/_base_client.py", line 1364, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/_base_client.py", line 1137, in request
    raise self._make_status_error_from_response(err.response) from None
anthropic.BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 207286 tokens > 200000 maximum'}, 'request_id': 'req_011CYw1JZbKcWtcfR9qP1qGA'}

```

### dogs_nonhierarchical (2026-03-11 08:50)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### seeds_stanified_model (2026-03-11 08:52)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### dugongs_model (2026-03-11 08:55)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### arK (2026-03-11 08:57)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### Mh_model (2026-03-11 08:59)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### lsat_model (2026-03-11 09:00)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### arma11 (2026-03-11 09:01)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### dugongs_model (2026-03-11 09:16)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### kronecker_gp (2026-03-11 09:16)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### nn_rbm1bJ10 (2026-03-11 09:17)
```
Exception: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 216487 tokens > 200000 maximum'}, 'request_id': 'req_011CYw3M8hS622kEecc3Weyr'}
Traceback (most recent call last):
  File "/home/user/daemon/projects/posteriordb/run_transpile.py", line 96, in <module>
    results[model_name] = transpile_model(model_name)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/posteriordb/run_transpile.py", line 65, in transpile_model
    result = transpile_stan_to_pymc(
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/pymc_rust_compiler/stan_to_pymc.py", line 341, in transpile_stan_to_pymc
    response = client.messages.create(
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/_utils/_utils.py", line 282, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/resources/messages/messages.py", line 996, in create
    return self._post(
           ^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/_base_client.py", line 1364, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/_base_client.py", line 1137, in request
    raise self._make_status_error_from_response(err.response) from None
anthropic.BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 216487 tokens > 200000 maximum'}, 'request_id': 'req_011CYw3M8hS622kEecc3Weyr'}

```

### nn_rbm1bJ100 (2026-03-11 09:18)
```
Exception: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 221870 tokens > 200000 maximum'}, 'request_id': 'req_011CYw3SsurAyNoH2SP5h4yV'}
Traceback (most recent call last):
  File "/home/user/daemon/projects/posteriordb/run_transpile.py", line 96, in <module>
    results[model_name] = transpile_model(model_name)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/posteriordb/run_transpile.py", line 65, in transpile_model
    result = transpile_stan_to_pymc(
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/pymc_rust_compiler/stan_to_pymc.py", line 341, in transpile_stan_to_pymc
    response = client.messages.create(
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/_utils/_utils.py", line 282, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/resources/messages/messages.py", line 996, in create
    return self._post(
           ^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/_base_client.py", line 1364, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/_base_client.py", line 1137, in request
    raise self._make_status_error_from_response(err.response) from None
anthropic.BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 221870 tokens > 200000 maximum'}, 'request_id': 'req_011CYw3SsurAyNoH2SP5h4yV'}

```

### ldaK2 (2026-03-11 09:18)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### ldaK5 (2026-03-11 09:18)
```
Exception: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 200395 tokens > 200000 maximum'}, 'request_id': 'req_011CYw3VLCTJWbQnoVxfxsuX'}
Traceback (most recent call last):
  File "/home/user/daemon/projects/posteriordb/run_transpile.py", line 96, in <module>
    results[model_name] = transpile_model(model_name)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/posteriordb/run_transpile.py", line 65, in transpile_model
    result = transpile_stan_to_pymc(
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/pymc_rust_compiler/stan_to_pymc.py", line 341, in transpile_stan_to_pymc
    response = client.messages.create(
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/_utils/_utils.py", line 282, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/resources/messages/messages.py", line 996, in create
    return self._post(
           ^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/_base_client.py", line 1364, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/daemon/projects/bayes-ai-compiler/.venv/lib/python3.11/site-packages/anthropic/_base_client.py", line 1137, in request
    raise self._make_status_error_from_response(err.response) from None
anthropic.BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 200395 tokens > 200000 maximum'}, 'request_id': 'req_011CYw3VLCTJWbQnoVxfxsuX'}

```

### irt_2pl (2026-03-11 09:19)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### hmm_example (2026-03-11 09:20)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### prophet (2026-03-11 09:22)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### hier_2pl (2026-03-11 09:22)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### gp_regr (2026-03-11 09:22)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```

### 2pl_latent_reg_irt (2026-03-11 09:24)
```
Validation failed: Agent did not achieve validation after 20 tool calls
```
