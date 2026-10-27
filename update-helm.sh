#!/bin/sh
# 새 이미지 태그를 kim-minji-helm에 반영하는 Merge Request를 만든다. MR 머지 = 배포 (ArgoCD는 helm 레포 main을 본다).
# HELM_REPO_TOKEN: kim-minji-helm에 push하고 MR을 만들 수 있는 GitLab 토큰 (api, write_repository 범위)
set -eu
CHART=ai-serving
GITLAB_URL=http://gitlab.local:8929
HELM_PROJECT=root%2Fkim-minji-helm
BRANCH="ci/${CHART}-${CI_COMMIT_SHORT_SHA}"

apk add --no-cache git curl
git config --global user.email "gitlab-ci@gitlab.local"
git config --global user.name "GitLab CI"
git clone --depth 1 "http://root:${HELM_REPO_TOKEN}@gitlab.local:8929/root/kim-minji-helm.git"
cd kim-minji-helm
API="${GITLAB_URL}/api/v4/projects/${HELM_PROJECT}"

# 재실행(같은 SHA) 대비: 브랜치가 이미 있으면 다시 만들지 않는다 (force-push 하지 않음)
if git ls-remote --exit-code --heads origin "$BRANCH" > /dev/null 2>&1; then
  echo "브랜치 ${BRANCH} 이미 있음 → 커밋·push 생략"
else
  git checkout -b "$BRANCH"
  # image.tag 줄(2칸 들여쓰기)만 바꾼다. 다른 'tag:' 문자열은 건드리지 않는다
  sed -i "s/^  tag:.*/  tag: \"${CI_COMMIT_SHORT_SHA}\"/" "${CHART}/values.yaml"
  if git diff --quiet; then
    echo "태그 변경 없음 → MR 생략"
    exit 0
  fi
  git commit -am "chore: ${CHART} 이미지 태그 ${CI_COMMIT_SHORT_SHA} 반영"
  git push origin "$BRANCH"
fi

# 같은 브랜치의 열린 MR이 있으면 새로 만들지 않는다
open_mrs=$(curl -sS --fail-with-body --header "PRIVATE-TOKEN: ${HELM_REPO_TOKEN}" \
  "${API}/merge_requests?state=opened&source_branch=${BRANCH}")
if [ "$open_mrs" != "[]" ]; then
  echo "MR 이미 열려 있음 (${BRANCH})"
  exit 0
fi

# 같은 차트의 MR이 여러 개 열려 있으면 가장 최신 것만 머지하고 나머지는 닫는다 (오래된 MR을 나중에 머지하면 이미지가 되돌아간다)
curl -sS --fail-with-body --request POST --header "PRIVATE-TOKEN: ${HELM_REPO_TOKEN}" \
  --data-urlencode "source_branch=${BRANCH}" \
  --data-urlencode "target_branch=main" \
  --data-urlencode "title=[${CHART}] 이미지 태그 ${CI_COMMIT_SHORT_SHA} 반영" \
  --data-urlencode "description=${CI_PROJECT_NAME} 커밋 ${CI_COMMIT_SHA} 빌드 이미지. 머지하면 ArgoCD가 배포한다. 같은 차트의 더 오래된 ci/ MR은 닫는다. 파이프라인: ${CI_PIPELINE_URL}" \
  "${API}/merge_requests"
echo
echo "MR 생성: ${BRANCH} → main"
