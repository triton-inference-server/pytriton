<!--
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Deployment Guide

This comprehensive guide covers deployment practices and configurations for PyTriton-based inference services in production environments. It includes:

- **Security Configuration** - Token-based access restriction and authentication
- **Container Deployment** - Docker containerization and configuration
- **Kubernetes Deployment** - Orchestration, health checks, and service exposure
- **Production Best Practices** - Security considerations and deployment patterns

## Secure Deployment Considerations

For comprehensive security deployment considerations and additional best practices, please refer to the [NVIDIA Triton Inference Server Secure Deployment Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/deploy.html).

### Token-Based Access Restriction

PyTriton provides built-in support for token-based access restriction to secure your model endpoints. This feature leverages Triton Inference Server's native security capabilities to protect sensitive endpoints from unauthorized access.

#### Overview

Token-based access restriction allows you to:

- **Protect sensitive endpoints** with authentication tokens
- **Control access** to model management and monitoring APIs
- **Prevent unauthorized access** to internal server functionality
- **Support both HTTP and gRPC protocols** with unified configuration

The security system automatically configures both HTTP and gRPC restrictions using the same token and endpoint configuration, ensuring consistent protection across all protocols.

#### Quick Start

##### Basic Usage with Auto-Generated Token

```python
from pytriton.triton import Triton
from pytriton.model_config import ModelConfig, Tensor
from pytriton.decorators import batch
import numpy as np

# Create Triton instance with token-based security enabled
# Token will be auto-generated if not provided
triton = Triton()

# Get the auto-generated access token for later use
access_token = triton.get_access_token()

@batch
def infer_fn(**inputs):
    # Your model inference logic here
    return {"output": inputs["input"] * 2}

triton.bind(
    model_name="secure_model",
    infer_func=infer_fn,
    inputs=[Tensor(name="input", dtype=np.float32, shape=(-1,))],
    outputs=[Tensor(name="output", dtype=np.float32, shape=(-1,))],
    config=ModelConfig(max_batch_size=8)
)

triton.run()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Client verification - test the deployed model with authentication
from pytriton.client import ModelClient
import numpy as np

# Test the deployed model
with ModelClient("localhost:8000", "secure_model", access_token=access_token) as client:
    # Create test input matching model specification
    test_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Perform inference
    result = client.infer_sample(input=test_input)

    # Verify output matches expected transformation (input * 2)
    expected_output = test_input * 2
    assert np.allclose(result["output"], expected_output)

# Cleanup code for codeblocks testing
triton.stop()
assert access_token is not None
```
-->

##### Using Explicit Access Token

```python
from pytriton.triton import Triton, TritonSecurityConfig

# Use your own access token with explicit security configuration
security_config = TritonSecurityConfig(access_token="my-secure-token-12345")

triton = Triton(security_config=security_config)
# Token is now set to your explicit value

# ... rest of your model setup
triton.run()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Cleanup code for codeblocks testing
triton.stop()
```
-->

#### Configuration Options

##### Default Protected Endpoints

By default, PyTriton protects these security-sensitive endpoints:

- `shared-memory` - Shared memory management
- `model-repository` - Model repository management
- `statistics` - Server statistics and metrics
- `trace` - Request tracing and debugging
- `logging` - Log level control

##### Custom Endpoint Protection

You can customize which endpoints to protect:

```python
from pytriton.triton import Triton, TritonSecurityConfig

# Protect only specific endpoints
custom_endpoints = ["statistics", "trace", "model-repository"]
security_config = TritonSecurityConfig(
    access_token="my-token",
    restricted_endpoints=custom_endpoints
)

triton = Triton(security_config=security_config)
# Only the specified endpoints will require token authentication
triton.run()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Cleanup code for codeblocks testing
triton.stop()
```
-->

##### Available Endpoints

PyTriton validates endpoint names against Triton Server's supported endpoints:

- `health` - Server health checks
- `metadata` - Server metadata
- `inference` - Model inference (usually not restricted)
- `shared-memory` - Shared memory operations
- `model-config` - Model configuration (required by ModelClient for initialization)
- `model-repository` - Model repository management
- `statistics` - Server statistics
- `trace` - Request tracing
- `logging` - Log level control

#### Security Best Practices

##### Token Management

1. **Use Strong Tokens**: When providing explicit tokens, use cryptographically secure random strings:

```python
import secrets
from pytriton.triton import Triton, TritonSecurityConfig

# Generate a secure token
secure_token = secrets.token_urlsafe(32)  # 256-bit security
security_config = TritonSecurityConfig(access_token=secure_token)

triton = Triton(security_config=security_config)
triton.run()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Cleanup code for codeblocks testing
triton.stop()
```
-->

2. **Store Tokens Securely**: Never hardcode tokens in source code. Use environment variables or secure configuration files:

```python
import os
from pytriton.triton import Triton, TritonSecurityConfig

# Load token from environment, if not set a random token will be generated
token = os.getenv("PYTRITON_ACCESS_TOKEN")
security_config = TritonSecurityConfig(access_token=token)

triton = Triton(security_config=security_config)
triton.run()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Cleanup code for codeblocks testing
triton.stop()
```
-->

3. **Rotate Tokens Regularly**: Implement token rotation policies for production deployments.

##### Endpoint Selection

1. **Protect Sensitive Operations**: Always protect endpoints that can modify server state or expose internal information:

```python
from pytriton.triton import Triton, TritonSecurityConfig

# Recommended for production
production_protected = [
    "shared-memory",
    "model-repository",
    "statistics",
    "trace",
    "logging"
]
security_config = TritonSecurityConfig(restricted_endpoints=production_protected)

triton = Triton(security_config=security_config)
triton.run()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Cleanup code for codeblocks testing
triton.stop()
```
-->

2. **Consider Inference Protection Carefully**: The `inference` endpoint is typically left open for normal model serving, but you may protect it if your use case requires access control for model inference:

```python
# Most common: Leave inference open for normal model serving
# security_config = TritonSecurityConfig(restricted_endpoints=["statistics", "trace"])

# Advanced use case: Protect inference for access-controlled model serving
# security_config = TritonSecurityConfig(restricted_endpoints=["inference", "statistics", "trace"])
```

3. **Consider Health Checks**: Be careful about restricting `health` endpoint if you use it for load balancer health checks.

4. **Model Configuration Access**: Be careful about restricting the `model-config` endpoint as PyTriton's ModelClient requires it for initialization. If you protect this endpoint, all ModelClient instances must include the access token:

```python
from pytriton.client import ModelClient

# If model-config is protected, clients need tokens
client = ModelClient("localhost:8000", "model_name", access_token="your-token")
```

##### Restoring Unrestricted Behavior

If you need to restore the previous unrestricted behavior (no token-based access restrictions), you can explicitly disable all restrictions:

```python
from pytriton.triton import Triton, TritonSecurityConfig

# Restore unrestricted behavior - no endpoints require tokens
security_config = TritonSecurityConfig(restricted_endpoints=[])

triton = Triton(security_config=security_config)
# All endpoints are accessible without tokens
# This restores the pre-v0.7.0 behavior
triton.run()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Cleanup code for codeblocks testing
triton.stop()
```
-->

**Important Distinctions:**

- `TritonSecurityConfig()` - Uses default protected endpoints (new secure behavior)
- `TritonSecurityConfig(restricted_endpoints=None)` - Same as above, uses defaults
- `TritonSecurityConfig(restricted_endpoints=[])` - **No restrictions** (unrestricted behavior)

**Migration Example:**

If you're upgrading from a version without token restrictions and want to maintain the same behavior:

```python
from pytriton.triton import Triton, TritonSecurityConfig

# Before v0.7.0 (no restrictions by default)
# triton = Triton()

# After v0.7.0 (to maintain same unrestricted behavior)
triton_unrestricted = Triton(security_config=TritonSecurityConfig(restricted_endpoints=[]))
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Cleanup code for codeblocks testing
triton_unrestricted.stop()
```
-->

#### Client Authentication

When endpoints are protected, clients need to provide the access token in their requests.

##### HTTP Clients

For HTTP requests, include the token in the `triton-access-token` header:

<!--
```python
from pytriton.triton import Triton, TritonSecurityConfig
from pytriton.model_config import ModelConfig, Tensor
from pytriton.decorators import batch
import numpy as np

@batch
def simple_infer(**inputs):
    return {"output": inputs["input"] * 2}

security_config = TritonSecurityConfig(access_token="your-access-token")
triton = Triton(security_config=security_config)
triton.bind(
    model_name="test_model",
    infer_func=simple_infer,
    inputs=[Tensor(name="input", dtype=np.float32, shape=(-1,))],
    outputs=[Tensor(name="output", dtype=np.float32, shape=(-1,))],
    config=ModelConfig(max_batch_size=8)
)

triton.run()
```
-->
<!--pytest-codeblocks:cont-->

```python
import requests

# Make authenticated request to protected endpoint
headers = {"triton-access-token": "your-access-token"}
response = requests.get("http://localhost:8000/v2/models/stats", headers=headers)
```

<!--pytest-codeblocks:cont-->
<!--
```python
try:
    assert response.status_code == 200
finally:
    triton.stop()
```
-->

##### PyTriton Clients

PyTriton's built-in clients automatically handle authentication when configured:

<!--
```python
from pytriton.triton import Triton, TritonSecurityConfig
from pytriton.model_config import ModelConfig, Tensor
from pytriton.decorators import batch
import numpy as np

@batch
def client_test_infer(**inputs):
    return {"output": inputs["input"] * 3}

# Start a Triton instance with security enabled for testing
security_config = TritonSecurityConfig(access_token="your-access-token")
triton = Triton(security_config=security_config)
triton.bind(
    model_name="model_name",
    infer_func=client_test_infer,
    inputs=[Tensor(name="input", dtype=np.float32, shape=(-1,))],
    outputs=[Tensor(name="output", dtype=np.float32, shape=(-1,))],
    config=ModelConfig(max_batch_size=8)
)
triton.run()
```
-->
<!--pytest-codeblocks:cont-->

```python
from pytriton.client import ModelClient

# Client will use the token for all endpoint access
client = ModelClient("http://localhost:8000", "model_name", access_token="your-access-token")
```

<!--pytest-codeblocks:cont-->
<!--
```python
try:
    test_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = client.infer_sample(input=test_input)
    expected_output = test_input * 3

    assert np.allclose(result["output"], expected_output)
finally:
    client.close()
    triton.stop()
```
-->

##### gRPC Clients

For gRPC clients, you need to include the token in the `triton-grpc-protocol-triton-access-token` header:

<!--
```python
from pytriton.triton import Triton, TritonSecurityConfig
from pytriton.model_config import ModelConfig, Tensor
from pytriton.decorators import batch
import numpy as np

@batch
def grpc_test_infer(**inputs):
    return {"output": inputs["input"] * 2}

# Start a Triton instance with security enabled for gRPC testing
security_config = TritonSecurityConfig(access_token="your-access-token")
triton = Triton(security_config=security_config)
triton.bind(
    model_name="model_name",
    infer_func=grpc_test_infer,
    inputs=[Tensor(name="input", dtype=np.float32, shape=(-1,))],
    outputs=[Tensor(name="output", dtype=np.float32, shape=(-1,))],
    config=ModelConfig(max_batch_size=8)
)
triton.run()
```
-->
<!--pytest-codeblocks:cont-->

```python
import tritonclient.grpc as grpcclient
from pytriton.client.auth import create_auth_headers
import numpy as np

# Create gRPC client
client = grpcclient.InferenceServerClient(url="localhost:8001")

# Create authentication headers for gRPC
access_token = "your-access-token"
headers = create_auth_headers(access_token, "grpc")

# Make authenticated request to protected endpoint
try:
    # Example: Get server metadata with authentication
    metadata = client.get_server_metadata(headers=headers)
    print("Server metadata retrieved successfully")

    # Example: Perform inference with authentication
    inputs = []
    inputs.append(grpcclient.InferInput("input", [1, 3], "FP32"))
    inputs[0].set_data_from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("output"))

    # Include headers in inference request
    result = client.infer("model_name", inputs, outputs=outputs, headers=headers)
    output_data = result.as_numpy("output")
    print(f"Inference result: {output_data}")

except Exception as e:
    print(f"Request failed: {e}")
finally:
    client.close()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Cleanup code for codeblocks testing
triton.stop()
```
-->

**Manual Header Creation:**

If you prefer to create headers manually without using the helper function:

<!--
```python
from pytriton.triton import Triton, TritonSecurityConfig
from pytriton.model_config import ModelConfig, Tensor
from pytriton.decorators import batch
import numpy as np

@batch
def grpc_test_infer(**inputs):
    return {"output": inputs["input"] * 2}

security_config = TritonSecurityConfig(access_token="your-access-token")
triton = Triton(security_config=security_config)
triton.bind(
    model_name="model_name",
    infer_func=grpc_test_infer,
    inputs=[Tensor(name="input", dtype=np.float32, shape=(-1,))],
    outputs=[Tensor(name="output", dtype=np.float32, shape=(-1,))],
    config=ModelConfig(max_batch_size=8)
)
triton.run()
```
-->
<!--pytest-codeblocks:cont-->

```python
import tritonclient.grpc as grpcclient

# Create gRPC client
client = grpcclient.InferenceServerClient(url="localhost:8001")

# Manually create gRPC authentication headers
access_token = "your-access-token"
headers = {"triton-grpc-protocol-triton-access-token": access_token}

# Use headers in requests
try:
    metadata = client.get_server_metadata(headers=headers)
    print("Server metadata retrieved successfully")
except Exception as e:
    print(f"Authentication failed: {e}")
finally:
    client.close()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Final cleanup code for codeblocks testing
triton.stop()
```
-->

**Important Notes for gRPC:**

- gRPC requires the `triton-grpc-protocol-` prefix before the header name
- The complete header name is `triton-grpc-protocol-triton-access-token`
- This differs from HTTP clients which use just `triton-access-token`
- Always include headers in both metadata requests and inference requests

#### Integration with Triton Server

PyTriton's token-based security integrates directly with Triton Inference Server's native security features:

- Uses `--grpc-restricted-protocol` for gRPC endpoint protection
- Uses `--http-restricted-api` for HTTP endpoint protection
- Follows Triton Server's standard token format: `<endpoints>:triton-access-token=<token>`
- Compatible with all Triton Server versions that support these features

#### Troubleshooting

##### Common Issues

1. **401/403 Unauthorized Responses**
   - Verify the token is correctly included in requests
   - Check that the endpoint is actually protected
   - Ensure token matches exactly (no extra spaces/characters)

2. **Endpoint Validation Errors**
   ```
   PyTritonValidationError: Invalid endpoint names: bad-endpoint. Valid endpoints are: ...
   ```
   - Check endpoint names against the supported list
   - Use hyphens, not underscores (e.g., `model-config` not `model_config`)

3. **Token Not Working**
   - Verify the server was started with token restrictions enabled
   - Check server logs for authentication errors
   - Ensure you're using the correct token from `get_access_token()`

4. **gRPC Authentication Issues**
   - **Wrong Header Format**: Ensure you're using `triton-grpc-protocol-triton-access-token`, not `triton-access-token`
   - **Missing Headers in Requests**: Include headers in both metadata and inference requests
   - **Protocol Mismatch**: Verify you're connecting to the gRPC port (usually 8001) not HTTP port (8000)

   ```python
   # ❌ Wrong - HTTP header format for gRPC
   headers = {"triton-access-token": "token"}

   # ✅ Correct - gRPC header format
   headers = {"triton-grpc-protocol-triton-access-token": "token"}

   # ✅ Or use the helper function
   from pytriton.client.auth import create_auth_headers
   headers = create_auth_headers("token", "grpc")
   ```

5. **Common gRPC Error Messages**
   - `StatusCode.UNAVAILABLE: This protocol is restricted, expecting header 'triton-grpc-protocol-triton-access-token'`
     → You're missing the authentication header or using wrong format
   - `StatusCode.UNAUTHENTICATED: Invalid access token`
     → The token is incorrect or has expired
   - `StatusCode.PERMISSION_DENIED: Access to this endpoint is restricted`
     → The endpoint requires authentication but no valid token was provided

##### Debugging

Enable verbose logging to see security configuration details:

```python
import logging
from pytriton.triton import Triton, TritonSecurityConfig

logging.basicConfig(level=logging.DEBUG)

security_config = TritonSecurityConfig(access_token="debug-token")
triton = Triton(security_config=security_config)
# Check what restrictions are configured
print("Server config:", triton._triton_server_config.to_cli_string())
triton.run()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Cleanup code for codeblocks testing
triton.stop()
```
-->

## Examples

### Production Deployment

```python
import os
import secrets
import logging
from pytriton.triton import Triton, TritonSecurityConfig
from pytriton.model_config import ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Production security configuration
def create_secure_triton():
    # Use environment token or generate secure one
    token = os.getenv("PYTRITON_ACCESS_TOKEN") or secrets.token_urlsafe(32)

    # Protect all sensitive endpoints
    protected_endpoints = [
        "shared-memory",
        "model-repository",
        "statistics",
        "trace",
        "logging"
    ]

    logger.info(f"Starting Triton with {len(protected_endpoints)} protected endpoints")

    security_config = TritonSecurityConfig(
        access_token=token,
        restricted_endpoints=protected_endpoints
    )

    return Triton(security_config=security_config)

# Usage
triton = create_secure_triton()


# Set up your models...
triton.run()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Cleanup code for codeblocks testing
triton.stop()
```
-->

### Development with Custom Endpoints

```python
from pytriton.triton import Triton, TritonSecurityConfig

# Development configuration - only protect statistics and tracing
dev_endpoints = ["statistics", "trace"]
security_config = TritonSecurityConfig(
    access_token="dev-token-123",
    restricted_endpoints=dev_endpoints
)

triton = Triton(security_config=security_config)

# Your model setup...
triton.run()
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Cleanup code for codeblocks testing
triton.stop()
```
-->

## API Reference

### Triton Class Parameters

- `security_config: Optional[TritonSecurityConfig] = None`
  - Security configuration object for token-based access restriction
  - If None, uses `DefaultTritonSecurityConfig` with auto-generated token and default protected endpoints

### TritonSecurityConfig Parameters

- `access_token: Optional[str] = None`
  - Access token for protected endpoints
  - If None, automatically generates a secure random token
  - Generated tokens are 32 characters long (256-bit security)

- `restricted_endpoints: Optional[List[str]] = None`
  - List of endpoint names to protect with token authentication
  - If None, uses default protected endpoints: `["shared-memory", "model-repository", "statistics", "trace", "logging"]`
  - If empty list `[]`, **disables all restrictions** (unrestricted behavior)
  - Valid endpoint names: `health`, `metadata`, `inference`, `shared-memory`, `model-config`, `model-repository`, `statistics`, `trace`, `logging`

**Behavior Summary:**
- `restricted_endpoints=None` → Use default protected endpoints (secure)
- `restricted_endpoints=[]` → No restrictions (unrestricted)
- `restricted_endpoints=["custom", "list"]` → Only specified endpoints protected

### Methods

- `get_access_token() -> str`
  - Returns the current access token (explicit or auto-generated)
  - Use this token for client authentication

### Constants

- `VALID_TRITON_ENDPOINTS: Set[str]`
  - Set of all valid Triton Server endpoint names
  - Used for endpoint validation

- `DEFAULT_PROTECTED_ENDPOINTS: List[str]`
  - Default list of security-sensitive endpoints to protect
  - Used when `restricted_endpoints=None`

For more details, see the [API Reference](../reference/triton.md).

## Container and Cluster Deployment

### How to deploy PyTriton in Docker containers

PyTriton can be packaged and deployed in Docker containers for consistent deployment across environments.

#### Expose required ports

When deploying in containers, expose the three essential ports:

```shell
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 your-pytriton-image
```

This exposes:
- Port 8000 for HTTP requests
- Port 8001 for gRPC requests
- Port 8002 for metrics collection

#### Configure shared memory size

PyTriton uses shared memory to pass data between Python callbacks and Triton Server. The default Docker shared memory (64MB) may be insufficient for large models.

Increase shared memory size based on your model requirements:

```shell
docker run --shm-size 8GB your-pytriton-image
```

Choose the shared memory size based on your largest expected batch size and tensor dimensions.

#### Set up container init process

Use Docker's init process to handle zombie process cleanup:

```shell
docker run --init your-pytriton-image
```

This ensures proper cleanup if PyTriton encounters unexpected errors.

### How to deploy PyTriton on Kubernetes

#### Configure container ports

Add port definitions to your Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pytriton-deployment
spec:
  template:
    spec:
      containers:
      - name: pytriton
        image: your-pytriton-image
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: grpc
        - containerPort: 8002
          name: metrics
```

#### Set up shared memory volume

Configure shared memory using emptyDir volume:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pytriton-deployment
spec:
  template:
    spec:
      volumes:
      - name: shared-memory
        emptyDir:
          medium: Memory
      containers:
      - name: pytriton
        image: your-pytriton-image
        volumeMounts:
        - mountPath: /dev/shm
          name: shared-memory
```

#### Configure health checks

Set up Kubernetes health checks using Triton's health endpoints:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pytriton-deployment
spec:
  template:
    spec:
      containers:
      - name: pytriton
        image: your-pytriton-image
        livenessProbe:
          httpGet:
            path: /v2/health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v2/health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Create service for external access

Expose PyTriton through a Kubernetes service:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: pytriton-service
spec:
  selector:
    app: pytriton
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: grpc
    port: 8001
    targetPort: 8001
  - name: metrics
    port: 8002
    targetPort: 8002
  type: LoadBalancer
```

## Additional Security Considerations

For comprehensive security deployment considerations beyond PyTriton's token-based access restriction, including:

- Deploying behind secure proxies and gateways
- Running with least privilege principles
- Container security best practices
- SSL/TLS configuration
- Network security considerations
- Resource access controls

Please refer to the official [NVIDIA Triton Inference Server Secure Deployment Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/deploy.html).
