# AWS / Amazon Bedrock

This guide creates a dedicated Amazon Bedrock API key for this benchmark and
uses it with the existing OpenAI-compatible Chat Completions adapter. Run these commands
yourself; they create AWS IAM resources and may incur model-inference charges.

Amazon Bedrock's `bedrock-runtime` endpoint accepts OpenAI Chat Completions
requests with a bearer API key, so no benchmark adapter changes are required.
The endpoint and model availability are regional, so use the same region
consistently throughout this guide.

The commands use the active AWS CLI profile and its configured default region.
Set those up first with the [AWS CLI configuration guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).

## What is required

To run the benchmark through Bedrock today, you need a region and compatible
model, an identity authorized to invoke that model, a Bedrock API key, and a
matching model configuration. The remaining steps marked
**recommended** improve isolation, diagnostics, or cleanup but can be skipped
when your existing AWS setup already provides their outcome.

## Step 1: Verify AWS access and select a model — required

```bash
aws sts get-caller-identity

aws bedrock list-foundation-models \
  --by-output-modality TEXT \
  --query 'modelSummaries[].{ID:modelId,Name:modelName,Streaming:inferenceTypesSupported}' \
  --output table
```

Choose an ID returned for your account and the CLI's default region, then set it:

```bash
export BENCH_MODEL_ID='replace-with-a-bedrock-model-id'
```

The selected model must support Bedrock Chat Completions and structured JSON
responses. Find capabilities in AWS's [Models at a glance](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html). Model IDs, supported APIs, quotas, and prices vary by region. Do not reuse an ID from another provider.

If the listing command is denied, an AWS administrator must grant Bedrock
permissions. A valid AWS CLI login alone does not imply Bedrock access.

## Step 2: Create a dedicated benchmark identity — recommended

Run this section only from an administrator identity allowed to create IAM
users and attach policies. It uses the AWS-managed `AmazonBedrockLimitedAccess`
policy recommended by Bedrock for API-key exploration. Use a dedicated user
instead of broadening an unrelated existing IAM user.

Skip this step if an existing, appropriately scoped identity already has the
required Bedrock permissions. Do not attach a broad policy to an unrelated
production identity merely to run this benchmark.

```bash
export BENCH_USER='jev-bench-bedrock'

aws iam get-user --user-name "$BENCH_USER" >/dev/null 2>&1 || \
  aws iam create-user --user-name "$BENCH_USER"

aws iam attach-user-policy \
  --user-name "$BENCH_USER" \
  --policy-arn arn:aws:iam::aws:policy/AmazonBedrockLimitedAccess
```

Your organization may use permission boundaries, SCPs, or custom policies.
The runtime identity needs Bedrock model-invocation permission and
`bedrock:CallWithBearerToken`. Creating a long-term key also requires
`iam:CreateServiceSpecificCredential`.

## Step 3: Create and hold the API key safely — required

This creates a service-specific credential that expires in seven days. AWS
returns the secret only once. The command captures it in the current shell
without printing it or writing it to the repository.

Bedrock allows at most two service-specific credentials for one user. First,
list existing Bedrock credentials. If two already exist, delete an unused one
before creating another; deleting it immediately invalidates that key.

```bash
aws iam list-service-specific-credentials \
  --user-name "$BENCH_USER" \
  --query 'ServiceSpecificCredentials[?ServiceName==`bedrock.amazonaws.com`].[ServiceSpecificCredentialId,CreateDate,Status]' \
  --output table
```

```bash
export BEDROCK_API_KEY="$(aws iam create-service-specific-credential \
  --user-name "$BENCH_USER" \
  --service-name bedrock.amazonaws.com \
  --credential-age-days 7 \
  --query 'ServiceSpecificCredential.ServiceCredentialSecret' \
  --output text)"

export BEDROCK_CREDENTIAL_ID="$(aws iam list-service-specific-credentials \
  --user-name "$BENCH_USER" \
  --query 'sort_by(ServiceSpecificCredentials[?ServiceName==`bedrock.amazonaws.com`], &CreateDate)[-1].ServiceSpecificCredentialId' \
  --output text)"

if [ -n "$BEDROCK_API_KEY" ] && [ "$BEDROCK_API_KEY" != "None" ]; then
  echo 'Bedrock API key captured.'
else
  echo 'Bedrock API key was not captured.'
fi
```

Never echo `BEDROCK_API_KEY`, put it in JSON, or commit it. If the shell closes,
retrieve it from an approved secret store or create a replacement: AWS cannot
show the key value again. The installed AWS CLI exposes this Bedrock value as
`ServiceCredentialSecret`; do not use `ServicePassword`.

For a short-lived alternative, generate a short-term API key from the Bedrock
console. It inherits the current IAM principal's permissions and expires with
the session, up to 12 hours. This is an optional, safer alternative to the
long-term key above. Using AWS credentials instead of a bearer key requires a
SigV4-capable benchmark adapter, which is not implemented yet.

## Step 4: Configure the benchmark — required

Create a local configuration. It names the environment variable but never
contains the API key.

```bash
cat > configs/bedrock.json <<EOF
{
  "adapter": "openai_compatible_chat_completions",
  "provider": "amazon-bedrock",
  "endpoint": "https://bedrock-runtime.$(aws configure get region).amazonaws.com/openai/v1/chat/completions",
  "api_key_env": "BEDROCK_API_KEY",
  "model_id": "${BENCH_MODEL_ID}",
  "model_revision": null,
  "timeout_seconds": 60,
  "max_retries": 2,
  "max_tokens": 1024,
  "response_format": "json_object",
  "model_config": {
    "temperature": 0,
    "reasoning_effort": "low"
  },
  "cost_basis": "unknown"
}
EOF

python3 -m json.tool configs/bedrock.json
```

For GPT-OSS through Bedrock Chat Completions, use `json_object` and
put its model-specific `reasoning_effort: low` argument inside
`model_config`. The adapter's explicit output contract and subsequent
validation still require a label from the frozen vocabulary and a numeric
confidence from 0 to 1. Use `json_schema` only for a selected model that
actually enforces it through this endpoint.

## Step 5: Verify one request before a benchmark run — recommended

This makes one paid model request. It verifies the endpoint, API key, model ID,
and basic Chat Completions access without starting the 3,080-item benchmark.

```bash
curl --fail-with-body --silent --show-error \
  --request POST "https://bedrock-runtime.$(aws configure get region).amazonaws.com/openai/v1/chat/completions" \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer $BEDROCK_API_KEY" \
  --data "{\"model\":\"${BENCH_MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the word ready.\"}]}" | python3 -m json.tool
```

After this succeeds, use the normal runner. It performs one synthetic
preflight and then continues into all 3,080 scored requests.

```bash
.venv/bin/jev-decision-bench run \
  --package "$PACKAGE" \
  --model-config configs/bedrock.json
```

## GPT-5.6 Luna through Responses

GPT-5.6 Luna supports the Responses API and structured outputs. In Tokyo,
Bedrock Runtime supports its global cross-Region profile, so use the separate
tracked configuration:

```bash
.venv/bin/jev-decision-bench run \
  --package "$PACKAGE" \
  --model-config configs/bedrock-openai-gpt-5.6-luna.json
```

The configuration uses the Responses endpoint and
`global.openai.gpt-5.6-luna`. Do not replace it with the in-Region model ID:
that ID is not available on Bedrock Runtime. The model card documents both the
[Responses API and structured-output support](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-luna.html).

## Step 6: Revoke and clean up — recommended after the experiment

When the experiment is complete, revoke the credential. If the dedicated IAM
user exists only for this benchmark, detach its policy and delete the user.

Keep the key and user only if you have a documented rotation and reuse plan.
Otherwise, cleanup is the safer default.

```bash
aws iam delete-service-specific-credential \
  --user-name "$BENCH_USER" \
  --service-specific-credential-id "$BEDROCK_CREDENTIAL_ID"

aws iam detach-user-policy \
  --user-name "$BENCH_USER" \
  --policy-arn arn:aws:iam::aws:policy/AmazonBedrockLimitedAccess

aws iam delete-user --user-name "$BENCH_USER"

unset BEDROCK_API_KEY BEDROCK_CREDENTIAL_ID
```

Do not delete the user if it has other policies, credentials, or workloads.

## Next

Read [Running experiments](running-experiments.md) to prepare the package and
run the benchmark with `configs/bedrock.json`.
