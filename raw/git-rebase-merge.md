# Git Rebase vs Merge

## 核心区别

Merge：保留完整的历史分支结构，创建一个merge commit。
Rebase：将分支的提交"移动"到目标分支的顶部，创造线性历史。

## Rebase的工作原理

1. 找到两个分支的共同祖先
2. 将当前分支的每个提交"重放"到目标分支之上
3. 生成新的commit（SHA不同）

## 黄金法则

**永远不要rebase公共分支**（main/master）。只rebase你自己的feature分支。

## 交互式Rebase

`git rebase -i HEAD~5` 可以：squash合并、reorder排序、edit修改、drop删除提交。

## 冲突处理

Rebase遇到冲突时，逐个提交解决，比merge更细致但也更繁琐。

## 实际建议

- 团队项目：feature分支用rebase保持整洁，公共分支用merge
- 个人项目：随意，rebase更清爽
- 已经push的分支：不要再rebase
