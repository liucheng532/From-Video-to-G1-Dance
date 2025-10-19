# 🔧 创建 WandB Registry 详细指南

## 🎯 问题说明

训练脚本报错：
```
HTTP 404: cannot create registry. custom registries can only be created via the registries ui
project 'wandb-registry-motions' not found
```

**原因**: WandB Registry 必须在网站上手动创建，不能通过代码自动创建。

---

## 📋 解决步骤

### 方式一：通过组织页面创建（推荐）

#### 1. 访问您的组织页面

在浏览器中打开：
```
https://wandb.ai/1393537481-the-hong-kong-university-of-science-and-techn
```

#### 2. 进入 Registry 页面

- 在左侧菜单中找到并点击 **"Registry"** 或 **"Registered Models"**
- 或者直接访问：
  ```
  https://wandb.ai/1393537481-the-hong-kong-university-of-science-and-techn/registry
  ```

#### 3. 创建新的 Collection

点击页面上的 **"Create new collection"** 或 **"New collection"** 按钮

#### 4. 填写 Collection 信息

在弹出的表单中填写：

| 字段 | 值 | 说明 |
|------|-----|------|
| **Collection name** | `Motions` | ⚠️ 必须准确填写为 "Motions" |
| **Description** | `Dance motion data for humanoid robots` | 可选，描述信息 |
| **Artifact type** | `dataset` | 选择 dataset 类型 |

#### 5. 确认创建

点击 **"Create"** 或 **"Create collection"** 按钮完成创建。

---

### 方式二：通过 Artifacts 页面创建

#### 1. 访问 Artifacts 页面

```
https://wandb.ai/1393537481-the-hong-kong-university-of-science-and-techn/artifacts
```

#### 2. 点击 Registry 相关选项

找到 **"Registry"** 标签或按钮

#### 3. 按照上述步骤 3-5 完成创建

---

## ✅ 验证 Registry 已创建

创建完成后，您应该能看到：

1. **Registry Collection 列表**中出现 "Motions"
2. **Collection 详情页面** URL 类似：
   ```
   https://wandb.ai/1393537481-the-hong-kong-university-of-science-and-techn/registry/Motions
   ```

---

## 🚀 创建完成后，重新运行训练

### 选项 A：使用重新训练脚本

```bash
./resume_train.sh
```

### 选项 B：手动执行命令

```bash
# 确保环境变量已设置
export WANDB_ENTITY=1393537481-the-hong-kong-university-of-science-and-techn

# 1. 转换舞蹈数据
python scripts/csv_to_npz.py \
  --input_file dance_motion/douyin1_g1.csv \
  --input_fps 30 \
  --output_name douyin1_dance \
  --robot g1 \
  --headless

# 2. 开始训练
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --registry_name ${WANDB_ENTITY}-org/wandb-registry-motions/douyin1_dance \
  --headless \
  --logger wandb \
  --log_project_name g1_dance_training \
  --run_name douyin1_v1 \
  --num_envs 4096
```

---

## 📸 界面参考

### 典型的 WandB Registry 页面布局

```
┌─────────────────────────────────────────────┐
│  [WandB Logo]  组织名                        │
├─────────────────────────────────────────────┤
│  Projects                                    │
│  Reports                                     │
│  Sweeps                                      │
│  Registry          ← 点击这里                │
│  Artifacts                                   │
│  ...                                         │
├─────────────────────────────────────────────┤
│                                              │
│  Registry Collections                        │
│                                              │
│  [+ Create new collection]  ← 点击这个按钮   │
│                                              │
└─────────────────────────────────────────────┘
```

### 创建表单示例

```
┌──────────────────────────────────────────┐
│  Create Collection                        │
├──────────────────────────────────────────┤
│  Collection name *                        │
│  ┌────────────────────────────────────┐  │
│  │ Motions                            │  │
│  └────────────────────────────────────┘  │
│                                           │
│  Description (optional)                   │
│  ┌────────────────────────────────────┐  │
│  │ Dance motion data for humanoid ... │  │
│  └────────────────────────────────────┘  │
│                                           │
│  Artifact type                            │
│  ┌────────────────────────────────────┐  │
│  │ dataset                        ▼   │  │
│  └────────────────────────────────────┘  │
│                                           │
│           [Cancel]    [Create]            │
└──────────────────────────────────────────┘
```

---

## ❓ 常见问题

### Q1: 找不到 Registry 菜单？

**解决方案**:
- 确认您有组织权限（不是个人账户）
- 尝试访问完整 URL：
  ```
  https://wandb.ai/1393537481-the-hong-kong-university-of-science-and-techn/registry
  ```
- 检查左侧导航栏，可能名为 "Registered Models" 或 "Model Registry"

### Q2: Collection name 必须是 "Motions" 吗？

**回答**: 是的，代码中硬编码了这个名称。如果想用其他名称，需要修改脚本。

### Q3: 创建后还是报错？

**检查**:
1. Collection 名称是否完全正确（区分大小写）
2. 刷新浏览器页面
3. 等待几秒钟让 WandB 同步
4. 重新运行 `./resume_train.sh`

### Q4: 如何查看已创建的 Registry？

访问：
```
https://wandb.ai/1393537481-the-hong-kong-university-of-science-and-techn/registry
```

应该能看到 "Motions" Collection。

---

## 📞 需要帮助？

如果创建 Registry 时遇到问题：

1. **检查权限**: 确保您是组织成员且有创建权限
2. **WandB 文档**: https://docs.wandb.ai/guides/artifacts/registry
3. **截图**: 如果看到的界面与描述不同，可以截图查看

---

## 🎉 下一步

创建完 Registry 后：

1. ✅ 运行 `./resume_train.sh` 重新开始训练
2. 📊 在 WandB 上查看训练进度
3. 🎮 训练完成后测试策略

---

**快速链接**:
- 您的组织: https://wandb.ai/1393537481-the-hong-kong-university-of-science-and-techn
- Registry 页面: https://wandb.ai/1393537481-the-hong-kong-university-of-science-and-techn/registry
- WandB 文档: https://docs.wandb.ai/guides/artifacts/registry







