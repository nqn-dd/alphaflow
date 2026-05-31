#!/bin/bash
# Run on qmcp-bastion via SSM. Deploys to EKS (qmcp-eks).
# Inputs (env vars set by deploy.yml):
#   APP_NAME           — e.g. faves-compliance
#   APP_NAMESPACE      — e.g. qmcp-services
#   MANIFEST_S3_KEY    — s3 key under qmcp-staging for rendered deployment manifest
#   NS_MANIFEST_S3_KEY — s3 key for namespace manifest
#   API_KEY_SECRET_ID  — AWS Secrets Manager id holding the API key (string) — optional
set -euxo pipefail

: "${APP_NAME:?}"
: "${APP_NAMESPACE:?}"
: "${MANIFEST_S3_KEY:?}"
: "${NS_MANIFEST_S3_KEY:?}"

# SSM root sessions don't set HOME → kubectl can't find kubeconfig. Force it.
export KUBECONFIG=/root/.kube/config
export AWS_REGION=us-east-1

cd /tmp

aws s3 cp "s3://qmcp-staging/${NS_MANIFEST_S3_KEY}" /tmp/ns.yaml --region "$AWS_REGION" --quiet
aws s3 cp "s3://qmcp-staging/${MANIFEST_S3_KEY}"    /tmp/app.yaml --region "$AWS_REGION" --quiet

# 1) Namespace (idempotent)
kubectl apply -f /tmp/ns.yaml

# 2) Sync API key secret from Secrets Manager → native K8s secret (idempotent)
if [ -n "${API_KEY_SECRET_ID:-}" ]; then
  API_KEY_VAL=$(aws secretsmanager get-secret-value \
    --secret-id "$API_KEY_SECRET_ID" --region "$AWS_REGION" \
    --query SecretString --output text)
  kubectl -n "$APP_NAMESPACE" create secret generic "${APP_NAME}-api-key" \
    --from-literal=api_key="$API_KEY_VAL" \
    --dry-run=client -o yaml | kubectl apply -f -
fi

# 3) Apply Deployment + Service + ServiceAccount
kubectl apply -f /tmp/app.yaml

# 4) Wait for rollout (12 min cap)
kubectl -n "$APP_NAMESPACE" rollout status deployment/"$APP_NAME" --timeout=12m

# 5) Report
kubectl -n "$APP_NAMESPACE" get deployment "$APP_NAME" -o wide
kubectl -n "$APP_NAMESPACE" get pods -l app="$APP_NAME" -o wide
