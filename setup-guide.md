# Git 使用指南

## 环境概览

本机有两个 Git 本地仓库，各自独立，互不影响。

| 本地仓库 | 远端仓库 | 分支 | 用途 |
|----------|----------|------|------|
| `/root/gittohub` | [github.com/lideolo/learnnote](https://github.com/lideolo/learnnote) | `main` | 学习笔记 / 文档 |
| `/root/RhythmMamba-main` | [github.com/lideolo/gitto](https://github.com/lideolo/gitto) | `main` | RhythmMamba 项目代码 |

## 账户配置

| 配置项 | 值 | 级别 |
|--------|-----|------|
| Git 用户名 | `lideolo` | 各仓库独立配置 |
| Git 邮箱 | `1909695763@qq.com` | 各仓库独立配置 |
| GitHub 账户 | [lideolo](https://github.com/lideolo) | - |

> 旧的全局配置 `RhythmMamba` / `rhythmmamba@example.com` 已清空。

## SSH 认证

- **密钥类型**: `ed25519`
- **私钥路径**: `~/.ssh/id_ed25519`
- **公钥路径**: `~/.ssh/id_ed25519.pub`
- **公钥内容**:
  ```
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICbgHzU721ROUwjOmJrCjpFFWArAXU9GuDaqAT9PQVJ/ 1909695763@qq.com
  ```
- **指纹**: `SHA256:dnW1dpPw6k2KsK6Esb+bdcQnpTP+elvCQZY+o0zi4Z4`
- **已添加**到 GitHub Settings → SSH and GPG keys

**测试连接**: `ssh -T git@github.com`（返回 "successfully authenticated" 即正常）

## Git 核心概念

### .git 目录

一个文件夹里只要有 `.git` 隐藏目录，它就是受 Git 管理的仓库。`.git` 里存放了所有历史记录、分支、配置等信息。

**VSCode 默认隐藏 `.git`，显示方法**: 设置 `files.exclude` → 把 `**/.git` 设为 `false`。

### Git 操作的是"当前目录"

Git 永远操作**你当前所在的目录**对应的仓库：

```bash
cd /root/gittohub
git status        # 操作 gittohub 仓库

cd /root/RhythmMamba-main
git status        # 操作 RhythmMamba 仓库
```

或者在任意目录加 `-C` 指定目标：

```bash
git -C /root/gittohub status
git -C /root/RhythmMamba-main status
```

> 两个仓库完全独立，互不影响。

### 配置级别

```bash
git config --global user.name "..."    # 全局生效，影响所有仓库
git config user.name "..."             # 仅当前仓库生效（推荐）
```

本机两个仓库均使用仓库级别配置。

## 数据流向

```
写代码 → git add → 暂存区 → git commit → 本地历史 → git push → GitHub
  ↑                                                            │
  └────────────────── git pull ─────────────────────────────────┘
```

## 日常操作流程

```bash
cd /root/gittohub              # 或 cd /root/RhythmMamba-main

git status                     # 第1步：查看哪些文件有变动

git add .                      # 第2步：标记所有变更，准备入库
git add 文件名                  # 或选择性添加

git commit -m "描述你改了什么"   # 第3步：创建本地快照

git push                       # 第4步：推送到 GitHub
```

### 操作频率建议

| 操作 | 频率 |
|------|------|
| `git add` + `git commit` | 每完成一个小修改就做一次 |
| `git push` | 每天结束或阶段性完成时 |

## 常见场景处理

### 新建仓库并推送

```bash
cd /root/新目录
git init
git config user.name "lideolo"
git config user.email "1909695763@qq.com"
echo "# 项目名" > README.md
git add README.md
git commit -m "Initial commit"
git remote add origin git@github.com:lideolo/仓库名.git
git branch -M main
git push -u origin main
```

### 远端仓库已有内容，本地也有内容（历史不相关）

```bash
git remote add origin git@github.com:lideolo/仓库名.git
git pull origin main --allow-unrelated-histories --no-rebase --no-edit

# 如果出现冲突，手动解决后：
git add .
git commit -m "合并远端"
git push -u origin main
```

### 修改远端地址

```bash
# 查看当前远端
git remote -v

# 修改 origin 的地址
git remote set-url origin git@github.com:lideolo/新仓库.git
```

### SSH vs HTTPS 远端

```bash
# SSH（推荐，免密）
git remote set-url origin git@github.com:lideolo/仓库名.git

# HTTPS（需要 Personal Access Token 认证）
git remote set-url origin https://github.com/lideolo/仓库名.git
```

### 查看提交历史

```bash
git log --oneline          # 简洁版
git log --oneline -5       # 最近5条
```

### 丢弃未提交的修改

```bash
git restore 文件名          # 丢弃单个文件的修改
git restore .               # 丢弃所有修改
```

## 注意事项

- **大文件**（>50MB）不要直接提交，应用 `.gitignore` 排除（如 `*.pth`、编译产物 `build/`）
- **敏感信息**（密码、token、密钥）绝不能提交到 Git
- **提交前先 `git status`** 确认要提交的文件是否正确
- **推送前先 `git pull`** 如果多人协作
