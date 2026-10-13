# 上传 GitHub 并启用每日运行

本次只修改了本地文件，还没有上传或启用自动任务。推荐创建一个单独的**私有仓库**，上传整个 `icir_tc_strategy` 目录，不要上传其父目录里的其他策略。

## 1. 登录 GitHub

打开 Mac 的“终端”，先进入项目：

```bash
cd "/Users/shc/Desktop/崇文量化/指数增强优化器/icir_tc_strategy"
gh auth login
```

当前电脑已安装 GitHub CLI (`gh`)。按提示选 GitHub.com、HTTPS、浏览器登录，按终端提示完成授权。不要把登录码、Token 或密码发进聊天。

```bash
gh auth setup-git
```

## 2. 创建私有仓库并上传

以下命令只适用于此目录**首次上传**。执行前确保你仍在上面的 `icir_tc_strategy` 目录：

```bash
git init -b main
git add .
git commit -m "Prepare daily paper strategy"
gh repo create icir-tc-strategy --private --source=. --remote=origin --push
gh repo view --web
```

最后两条会创建远端仓库并上传，浏览器会打开仓库主页。若同名仓库已存在，应换一个未使用的仓库名；不要删除已有仓库。

若 `git commit` 提示身份未知，先在**本项目**设置你自己的提交姓名、邮箱，再重试失败的那条命令：

```bash
git config user.name "你的名字"
git config user.email "你的GitHub提交邮箱"
```

提交邮箱可使用 GitHub Settings → Emails 中的 noreply 地址。浏览器拖拽上传容易遗漏隐藏的 `.github` 目录，不推荐用来上传本项目。

## 3. 先做一次手动云端验证

1. 仓库 **Settings → Actions → General**：确认允许运行本项目用到的 Actions。工作流已声明 `contents: write`，如组织策略限制，需由仓库/组织管理员允许工作流写入此仓库。默认分支保护若禁止 bot 直接写入，同样会使保存失败。
2. 仓库 **Actions**：首次上传会运行 `ICIR-TC tests`，应全部通过。
3. 在 A 股**交易日北京时间 16:00 后**，选择 **ICIR-TC daily run → Run workflow → main → Run workflow**。建议 17:10 后数据更稳定时试跑。
4. 首次需要刷新行业和下载约 300 只股票的历史日线及当日未复权校验行情，耗时取决于行情接口。任务正在下载时无需反复触发。
5. 等待任务完成，核对下面的验收标准。休市日或 16:00 前绿色结束仅代表跳过，不能算全链路验收。

**首次验收必须同时满足：**

- 任务页面测试、运行、持久化步骤均成功；日志最后有 `[done]`。
- 仓库出现当天 `reports/daily/日期.md`。
- `state/book.json` 的 `meta.last_run` 为当天，`initialized` 为 `true`，持仓非空，权重合计约为 1。
- 当天调仓报告求解成功、成本已扣除，并存在 `_trades.csv`。若优化失败、仍保持现金，即便任务完成也不能算成功建仓。
- 仓库提交记录出现 bot 的 `run: save paper ledger and reports ...`。
- 再点一次 Run workflow，日志显示已记账跳过，净值和交易费用不再变化。

本地离线测试使用模拟行情验证逻辑；尚未在你的 GitHub 账户执行全量行情任务。云端网络可用性以这次试跑为准。

## 4. 打开每日自动运行

手动验收通过后，在仓库：

**Settings → Secrets and variables → Actions → Variables → New repository variable**

填写：

```text
Name: ENABLE_DAILY
Value: true
```

注意是 **Variables**，不是 Secrets；值是小写 `true`。

此后每个周一至周五北京时间 **17:10** 和 **18:30** 尝试运行。程序识别 A 股节假日；第二次在当天已记账时跳过。你关机不影响云端运行。

GitHub cron 使用 UTC，任务可能排队、延迟甚至漏触发；程序会在下次成功运行补齐估值，漏掉的调仓顺延到恢复日。定时工作流须在仓库默认分支上。私有仓库运行分钟数和 Artifact 存储受账户套餐/预算限制，查看 Settings → Billing 的使用情况。

暂停时删除 `ENABLE_DAILY` 或改成 `false`。这只暂停定时执行，手动 Run workflow 仍可运行。

## 5. 查看日报、调仓与持续维护

- 日报：仓库 **Code → reports → daily**，点击最新日期 `.md`。
- 月度调仓：**reports → rebalance**，报告看诊断，CSV 看完整目标权重。
- 当前组合和累计净值：`state/book.json`，或下载对应 Parquet 副本进行分析。
- 运行错误：**Actions → 对应任务 → 红色步骤**；页面底部的 **Artifacts** 有恢复包，配置保留 30 天。

修改策略前先同步云端最新账本：

```bash
cd "/Users/shc/Desktop/崇文量化/指数增强优化器/icir_tc_strategy"
git pull --ff-only
```

若提示冲突，先保留双方文件并核对，不使用强制覆盖。不要手工修改处理日期来“补跑”，也不要删除账本后让程序不知情地重新建仓。研究策略切版本建议新仓库、新账本，避免将不同版本拼成同一绩效。

## 常见失败与恢复

| 情况 | 含义和处理 |
|---|---|
| AKShare 下载失败/超时/字段变化 | 当次不提交账本；稍后重试。若持续失败，检查数据源或云端网络，不能通过放松数据检查掩盖问题 |
| 行情缺失、停牌缺日、历史不足 | 程序暂停而不是按零收益处理；需要可靠的停牌/复牌或补充行情数据后再恢复 |
| 优化不可行 | 原组合继续估值，报告说明，下个交易日重试。若持续失败，应审查约束与输入，不自动改成指数持仓 |
| 行业更新失败 | 保留旧缓存且停止本次云任务；稍后重试，成功后自动保存更新 |
| `git push` 被拒 | 算出的文件尚未持久化。先下载该任务 Artifact，检查写权限、分支保护或并发修改；不要强推 |
| Artifact 有文件但任务失败 | 文件可能只写入了一部分；`state/book.json` 才是权威账本。不能把 Artifact 中的日报当作已入账凭证 |

推送失败后若要恢复 Artifact，先暂停定时，保存当前仓库和恢复包，比较两边 `book.json` 的最后日期及持仓；只有确认远端尚未包含这次运行且没有更新的账本，才可整体恢复匹配的 state/reports 并正常提交。遇到不确定的冲突应核对后处理，不拼接两个账本。首次运行在写盘时中断若留下 Parquet 却没有 book.json，程序会主动拒绝作为旧账本加载；保留该包后人工核对重建。

## 相关官方说明

- [GitHub CLI 创建仓库](https://cli.github.com/manual/gh_repo_create)
- [定时工作流的限制](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)
- [仓库 Actions 设置](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository)
