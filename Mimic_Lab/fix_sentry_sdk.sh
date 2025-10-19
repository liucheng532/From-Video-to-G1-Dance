#!/bin/bash
# 修复 sentry-sdk 版本冲突（可选）

echo "回退 sentry-sdk 到兼容版本..."
pip install sentry-sdk==1.43.0

echo "✓ 修复完成"







