# Git 仓库初始化与推送指南

## 环境信息

| 项目 | 值 |
|------|-----|
| 本地目录 | `/root/gittohub` |
| 远端仓库 | [github.com/lideolo/gitto](https://github.com/lideolo/gitto) |
| 分支 | `main` |

## 账户配置

| 配置项 | 值 | 级别 |
|--------|-----|------|
| Git 用户名 | `lideolo` | 仓库级别 |
| Git 邮箱 | `1909695763@qq.com` | 仓库级别 |
| GitHub 账户 | [lideolo](https://github.com/lideolo) | - |

## SSH 认证

- **密钥类型**: `ed25519`
- **私钥路径**: `~/.ssh/id_ed25519`
- **公钥路径**: `~/.ssh/id_ed25519.pub`
- **公钥内容**:
  ```
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICbgHzU721ROUwjOmJrCjpFFWArAXU9GuDaqAT9PQVJ/ 1909695763@qq.com
  ```
- **指纹**: `SHA256:dnW1dpPw6k2KsK6Esb+bdcQnpTP+elvCQZY+o0zi4Z4`

## 操作步骤回顾

### 1. 清空旧的全局 Git 配置

```bash
git config --global --unset user.name
git config --global --unset user.email
```

### 2. 初始化本地仓库

```bash
cd /root/gittohub
git init
```

### 3. 配置仓库级别用户信息

```bash
git config user.name "lideolo"
git config user.email "1909695763@qq.com"
```

### 4. 创建初始文件并提交

```bash
echo "# gitto" > README.md
git add README.md
git commit -m "Initial commit"
```

### 5. 生成 SSH 密钥

```bash
ssh-keygen -t ed25519 -C "1909695763@qq.com" -f ~/.ssh/id_ed25519 -N ""
```

### 6. 添加公钥到 GitHub

在 GitHub → Settings → SSH and GPG keys → New SSH Key 中添加公钥。

### 7. 添加远端并推送

```bash
git remote add origin git@github.com:lideolo/gitto.git
git branch -M main
git push -u origin main
```

## 日常使用

```bash
cd /root/gittohub
git add .
git commit -m "你的提交信息"
git push
```
