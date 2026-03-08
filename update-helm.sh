#!/bin/sh
apk add --no-cache git
git config --global user.email "gitlab-ci@gitlab.local"
git config --global user.name "GitLab CI"
git config --global http.sslVerify false
git clone http://root:${HELM_REPO_TOKEN}@gitlab.local:8929/root/kim-minji-helm.git
cd kim-minji-helm
sed -i "s/tag:.*/tag: \"${CI_COMMIT_SHORT_SHA}\"/" ai-serving/values.yaml
git add ai-serving/values.yaml
git commit -m "ci update ai-serving tag ${CI_COMMIT_SHORT_SHA}"
git push origin main