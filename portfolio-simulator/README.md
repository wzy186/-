# 投资组合分析与调仓模拟

度小满全栈开发实习技术测评项目（VC-FIN-001）。基于固定行情快照的单用户组合分析原型：组合概览、集中度检查（单一标的 >30%、行业 >50% 触发，等于不触发）、顺序调仓模拟与原/模拟组合对比、方案命名保存与重启后重开。扩展功能：**目标权重调仓建议**（等权分散/自定义权重，后端生成整数股数建议并经与人工操作相同的引擎校验，附目标/实际偏差报告）、**方案删除**（二次确认）与**方案载入模拟器派生草稿**（原方案只读不变）。

## 文档导航（评审阅读路径）

| 评审关注点 | 位置 |
|---|---|
| 需求拆解：FR 映射、业务规则与假设、实现范围取舍 | `PLAN.md`（需求理解/待确认问题与假设/最小范围）+ `DESIGN.md` §1（FR→实现→验证映射、功能边界、状态交互） |
| 数据与状态设计：实体关系、原始组合隔离、保存内容与重算策略 | `DESIGN.md` §2 |
| 系统结构与接口设计：结构图、接口表、错误语义、前后端口径统一 | `DESIGN.md` §3 |
| 计算与异常：金额精度方案、阈值边界、验证方法、异常设计 | `DESIGN.md` §4 |
| 设计决策与取舍（7 项，含备选方案） | `DESIGN.md` §5 |
| AI 生成代码与计算结果的核验（3 个真实协作片段，含一次通过测试发现错误的实际核验） | `DESIGN.md` §7 + `VERIFY.md` |
| 完成情况、未完成项、与初始方案差异、复盘 | `DESIGN.md` §6 |
| 运行、验收复现步骤与验收用例执行记录 | 本 README + `VERIFY.md` §2 |

## 技术栈与环境

| 组件 | 版本（实测） | 说明 |
|---|---|---|
| 后端 | Python 3.13.5 + FastAPI + sqlite3（标准库） | 依赖锁定见 `backend/requirements.txt` |
| 前端 | React 18 + Vite 5 + TypeScript（已构建，`frontend/dist/` 随包提供） | Node ≥18 仅重建时需要 |
| 存储 | SQLite 文件（`backend/portfolio.db`，首次启动自动创建并写入种子快照） | |
| 数据库 | — | 无外部服务依赖 |

支持 macOS / Linux / Windows（Python 3.10+）。设计文档见 `DESIGN.md`，验证记录见 `VERIFY.md`。

## 快速开始（推荐：无需 Node）

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器访问 **http://127.0.0.1:8000**（前端已构建并随包提供，由后端直接托管）。

## 重新构建前端（可选）

```bash
cd frontend
npm install          # 若 npm 镜像超时(504)，重试一次即可
npm run build        # 产物输出到 frontend/dist/，后端自动托管新版本
```

## 开发模式（前后端热更新）

```bash
# 终端 1：后端
cd backend && .venv/bin/uvicorn app.main:app --port 8000
# 终端 2：前端（/api 自动代理到 8000）
cd frontend && npm install && npm run dev   # 访问 http://localhost:5173
```

## 方案重开步骤（FR-04 / AC-05）

1. 「调仓模拟」页添加多笔操作（如：卖出 A01 × 100，再买入 A02 × 100），输入方案名称，点击「保存方案」；
2. 切换到「已存方案」页，点击方案的「打开」查看完整对比结果（只读）；
3. 刷新页面、或 Ctrl+C 重启后端后再次打开：模拟结果不变（保存于 SQLite），原始组合始终为现金 20,000.00 元、A01 400 股、A02 400 股。

## 运行测试

```bash
cd backend && .venv/bin/pytest -q        # 67 用例：引擎单元测试 + AC-01~05 API 验收测试 + 调仓建议/方案管理测试
```

## 验收辅助 URL 参数

| 示例 | 作用 |
|---|---|
| `/?tab=overview` / `simulate` / `plans` | 指定初始页签 |
| `/?ops=SELL:A01:100,BUY:A02:100` | 预填操作列表并进入模拟页（对应 AC-03） |
| `/?ops=BUY:A01:201` | 复现 AC-04 现金不足错误反馈 |
| `/?plan=1` | 打开已保存的方案 1（对应 AC-05） |
| `/?tab=simulate&rebalance=equal` | 自动生成等权分散调仓建议（扩展功能截图） |

## 目录结构

```
PLAN.md / DESIGN.md / README.md / VERIFY.md   设计文档与验证记录
data/投资组合示例数据.json                      种子数据（行情快照）
backend/app/                                    FastAPI 应用（engine 纯函数引擎 / money 精度 / db / routers）
backend/tests/                                  pytest 测试（AC-01~05 映射）
frontend/src/                                   React 源码（组件化，深浅双模式）
scripts/screenshots.sh                          验收截图生成脚本
docs/screenshots/                               验收截图
```

## 提交说明

- **实际投入时长**：约 1.3 小时（含设计与验证；AI 辅助开发压缩了编码时间，详见 DESIGN.md §7）
- **AI 工具**：Claude Code（需求拆解、方案设计、代码与测试生成、文档草拟；人工确认与验证）
- **未完成项 / 已知问题**：见 DESIGN.md §6（多用户与并发控制未做——明确超出单用户范围；无影响运行的未完成项）
- 数据均为虚构测评数据，不构成投资建议。
