# 重命名 GitHub 仓库步骤

## 步骤 1: 在 GitHub 上重命名仓库

1. 访问你的仓库：https://github.com/PrecipiceBlades/10-414-hw1
2. 点击 **Settings** 标签
3. 向下滚动到 **Repository name** 部分
4. 将 `10-414-hw1` 改为 `10414-f22`
5. 点击 **Rename** 按钮
6. GitHub 会自动重定向到新名称的仓库

## 步骤 2: 更新本地仓库的远程 URL

在本地仓库目录中运行：

```bash
# 更新远程 URL
git remote set-url origin git@github.com:PrecipiceBlades/10414-f22.git

# 验证更改
git remote -v
```

## 步骤 3: (可选) 重命名本地目录

如果你想重命名本地目录：

```bash
# 1. 先回到父目录
cd /root

# 2. 重命名目录
mv 10-414-hw1 10414-f22

# 3. 进入新目录
cd 10414-f22
```

## 注意事项

- GitHub 会自动处理旧 URL 的重定向（通常保留一段时间）
- 所有现有的克隆和 fork 仍然可以工作（通过重定向）
- 但建议更新所有本地克隆的远程 URL
- 如果仓库有 collaborators，通知他们更新远程 URL

## 验证

运行以下命令验证一切正常：

```bash
git remote -v
git fetch origin
```

应该显示新的远程 URL，并且 `git fetch` 应该成功。

