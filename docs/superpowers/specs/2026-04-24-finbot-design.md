# Finbot — AI 财务助手系统设计文档

**日期：** 2026-04-24  
**项目目录：** `d:/program/VSCodeW/Finbot`  
**定位：** 小团队 / 企业级 AI 财务助手，支持自然语言记账、多用户权限、智能预算预警

---

## 1. 项目概述

Finbot 是一个 AI 驱动的财务管理系统，用户通过自然语言与系统对话完成交易记录、报告生成和预算管理。系统支持多用户团队协作，具备多轮上下文理解能力。

**核心能力：**
- 自然语言 → 结构化交易记录（OpenAI Function Calling）
- 多步 LangGraph Agent 工作流：记录 → 分类 → 报告 → 预算预警
- Redis 多轮会话记忆，降低重复 token 消耗
- 小团队多用户权限体系（owner / admin / member / viewer）
- JWT + OAuth 2.0 混合认证（本地账号 + Google / GitHub）

---

## 2. 技术选型

| 层次 | 技术 |
|------|------|
| 前端 | React + TypeScript + Tailwind CSS |
| 后端 API | FastAPI (Python) |
| AI 工作流 | LangGraph + OpenAI Function Calling |
| 异步任务 | Celery + Redis (Broker + Result Backend) |
| 会话记忆 | Redis（TTL 24h） |
| 主数据库 | PostgreSQL + SQLAlchemy ORM + Alembic |
| 认证 | JWT (python-jose) + OAuth 2.0 (authlib) |
| 实时推送 | Server-Sent Events (SSE) |

---

## 3. 系统架构

采用**异步任务队列式**架构（方案 B），API 层与 AI 任务执行层解耦。

```
React Frontend (TypeScript)
        ↕ HTTPS / SSE
   FastAPI Gateway
   (REST + SSE + JWT/OAuth 认证)
        ↕ 任务分发 / 结果回写
Redis (Celery Broker)  →  Celery Workers (LangGraph Agent)
        ↕ 读写
PostgreSQL  |  OpenAI API  |  OAuth Provider (Google/GitHub)
```

**核心设计原则：**
- FastAPI 收到自然语言请求后，立即返回 `task_id`，不阻塞 HTTP 连接
- 前端通过 `GET /chat/stream/{task_id}` SSE 端点订阅 Agent 执行进度
- Redis 身兼两职：Celery 任务队列 + LangGraph 会话记忆存储
- Celery Worker 数量可水平扩展，不影响 API 层响应速度

---

## 4. LangGraph Agent 工作流

用户每次发送消息触发一个完整的 Agent 执行任务（Celery task）：

```
① 用户输入 + 加载 Redis 会话记忆 (conversation_id → history)
         ↓
② 意图识别：OpenAI Function Calling
   定义 functions: record_transaction / generate_report / clarify
         ↓ 根据调用的 function 路由
    ┌──────────────────────────────────────────┐
    │ record_transaction  │ generate_report    │ clarify
    │ ③-A 写入交易记录    │ ③-B 聚合+生成摘要  │ ③-C 追问用户
    │ PostgreSQL INSERT   │ OpenAI 摘要→存储   │ 保留上下文等待
    └──────────────────────────────────────────┘
         ↓ (③-A 路径继续)
④ 预算预警检查
   查询当月该类别累计支出 vs 预算 alert_threshold
   超阈值 → 写入 alerts 表 + SSE 实时推送前端
         ↓
⑤ 更新 Redis 会话记忆 + SSE 推送最终响应
```

**Function Calling Schema（关键字段）：**

> AI 输出金额单位为"元（yuan）"，后端服务层统一乘以 100 转为"分"再写库。

`record_transaction`:
- `amount_yuan` (number) — 金额，单位"元"（如 15.5），后端转为 fen 存储
- `direction` (income | expense)
- `category` (string) — AI 推断的分类名
- `account_name` (string)
- `transaction_date` (ISO 8601)
- `description` (string) — 原始描述

`generate_report`:
- `period_start`, `period_end` (ISO 8601)
- `group_by` (category | account | day)

---

## 5. 数据模型（PostgreSQL）

> **金额存储约定：** 所有金额字段使用 `BIGINT`，单位为"分"（fen）。  
> 示例：¥1.50 存储为 `150`。展示层除以 100，API 文档需明确说明单位。

### 核心表结构

**users**
```
id          UUID PRIMARY KEY
email       VARCHAR UNIQUE NOT NULL
password_hash VARCHAR (OAuth 登录时为 NULL)
oauth_provider VARCHAR (google | github | null)
oauth_id    VARCHAR
name        VARCHAR
avatar_url  VARCHAR
is_active   BOOLEAN DEFAULT true
created_at  TIMESTAMP
updated_at  TIMESTAMP
```

**teams**
```
id          UUID PRIMARY KEY
name        VARCHAR NOT NULL
owner_id    UUID REFERENCES users(id)
plan        VARCHAR DEFAULT 'free'
created_at  TIMESTAMP
```

**team_members**
```
id          UUID PRIMARY KEY
team_id     UUID REFERENCES teams(id)
user_id     UUID REFERENCES users(id)
role        VARCHAR (owner | admin | member | viewer)
joined_at   TIMESTAMP
UNIQUE (team_id, user_id)
```

**accounts**
```
id          UUID PRIMARY KEY
team_id     UUID REFERENCES teams(id)
name        VARCHAR NOT NULL
type        VARCHAR (bank | credit | cash | investment)
currency    VARCHAR DEFAULT 'CNY'
balance_fen BIGINT DEFAULT 0
is_active   BOOLEAN DEFAULT true
created_at  TIMESTAMP
```

**categories**
```
id          UUID PRIMARY KEY
team_id     UUID REFERENCES teams(id) -- NULL 表示系统默认分类
name        VARCHAR NOT NULL
icon        VARCHAR
parent_id   UUID REFERENCES categories(id) -- 支持多级分类
```

**transactions（核心表）**
```
id               UUID PRIMARY KEY
team_id          UUID REFERENCES teams(id)
account_id       UUID REFERENCES accounts(id)
category_id      UUID REFERENCES categories(id)
amount_fen       BIGINT NOT NULL       -- 单位：分
direction        VARCHAR (income | expense)
description      TEXT                  -- 原始自然语言
transaction_date DATE NOT NULL
created_by       UUID REFERENCES users(id)
created_at       TIMESTAMP
INDEX (team_id, transaction_date)
INDEX (team_id, category_id)
```

**budgets**
```
id               UUID PRIMARY KEY
team_id          UUID REFERENCES teams(id)
category_id      UUID REFERENCES categories(id)
amount_fen       BIGINT NOT NULL       -- 月度/季度预算，单位：分
period           VARCHAR (monthly | quarterly)
alert_threshold  FLOAT DEFAULT 0.8    -- 0.0~1.0，触发预警比例
is_active        BOOLEAN DEFAULT true
created_at       TIMESTAMP
```

**alerts**
```
id               UUID PRIMARY KEY
team_id          UUID REFERENCES teams(id)
budget_id        UUID REFERENCES budgets(id)
triggered_by     UUID REFERENCES transactions(id)
usage_ratio      FLOAT                 -- 触发时的消耗比例
message          TEXT                  -- AI 生成的预警文本
is_read          BOOLEAN DEFAULT false
created_at       TIMESTAMP
```

**reports**
```
id               UUID PRIMARY KEY
team_id          UUID REFERENCES teams(id)
title            VARCHAR
period_start     DATE
period_end       DATE
content          TEXT      -- AI 生成的 Markdown 摘要
raw_data         JSONB     -- 聚合数值，避免重复查询
created_by       UUID REFERENCES users(id)
created_at       TIMESTAMP
```

---

## 6. API 设计

### 认证

```
POST /auth/register                    本地注册（email + password）
POST /auth/login                       本地登录 → 返回 access_token + refresh_token
GET  /auth/oauth/{provider}            跳转 OAuth 授权页
GET  /auth/oauth/callback              OAuth 回调 → 签发 JWT
POST /auth/refresh                     使用 refresh_token 换新 access_token
```

### 对话 & Agent 任务

```
POST /chat/message                     发送自然语言消息 → 返回 { task_id }
GET  /chat/stream/{task_id}            SSE 订阅 Agent 执行进度（流式）
GET  /chat/history?page=1&size=20      会话历史分页
DELETE /chat/history                   清空当前用户会话记忆
```

### 财务数据

```
GET    /transactions                   列表（过滤：date_from/to, category_id, account_id）
POST   /transactions                   手动创建（绕过 AI）
GET    /transactions/{id}              详情
DELETE /transactions/{id}              删除

GET    /accounts                       账户列表
POST   /accounts                       创建账户
PATCH  /accounts/{id}                  更新账户

GET    /categories                     分类列表（含系统默认 + 团队自定义）
POST   /categories                     创建自定义分类

GET    /budgets                        预算列表
POST   /budgets                        配置预算
PATCH  /budgets/{id}                   更新预算
DELETE /budgets/{id}                   删除预算

GET    /alerts?is_read=false           预警通知列表
PATCH  /alerts/{id}/read               标为已读

GET    /reports                        历史报告列表
GET    /reports/{id}                   报告详情
```

### 团队 & 权限

```
POST   /teams                          创建团队
GET    /teams/{id}                     团队信息
GET    /teams/{id}/members             成员列表
POST   /teams/{id}/members             邀请成员（发送邀请邮件）
PATCH  /teams/{id}/members/{uid}/role  变更成员角色
DELETE /teams/{id}/members/{uid}       移除成员
```

### 统一错误响应格式

```json
{
  "error": {
    "code": "TRANSACTION_NOT_FOUND",
    "message": "指定交易记录不存在",
    "details": { "transaction_id": "uuid-xxx" }
  }
}
```

错误码命名空间：
- `AUTH_*` — 认证/授权相关
- `TRANSACTION_*` — 交易记录相关
- `AGENT_*` — LangGraph Agent 执行错误
- `PERMISSION_*` — 权限不足
- `VALIDATION_*` — 请求参数校验失败

---

## 7. 权限控制

| 操作 | owner | admin | member | viewer |
|------|-------|-------|--------|--------|
| 查看交易/报告 | ✓ | ✓ | ✓ | ✓ |
| 创建/编辑交易 | ✓ | ✓ | ✓ | ✗ |
| 删除交易 | ✓ | ✓ | ✗ | ✗ |
| 管理预算/账户 | ✓ | ✓ | ✗ | ✗ |
| 邀请/移除成员 | ✓ | ✓ | ✗ | ✗ |
| 变更成员角色 | ✓ | ✗ | ✗ | ✗ |
| 删除团队 | ✓ | ✗ | ✗ | ✗ |

权限检查在 FastAPI 依赖注入层（`Depends`）中统一处理，不散落在业务逻辑中。

---

## 8. Redis 数据结构

**会话记忆（LangGraph）：**
```
Key:   session:{user_id}:{conversation_id}
Type:  List (JSON 序列化的消息对象)
TTL:   86400s (24h)
Value: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
```

**Celery 任务状态：**
```
Key:   celery-task-meta-{task_id}
Type:  String (Celery 自动管理)
TTL:   3600s (1h，任务完成后)
```

**SSE 进度推送（可选 Pub/Sub）：**
```
Channel: task-progress:{task_id}
Message: {"step": "function_calling", "status": "running", "message": "正在解析..."}
```

---

## 9. 项目目录结构（建议）

```
Finbot/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI 路由（auth, chat, transactions, teams...）
│   │   ├── agent/        # LangGraph 工作流节点定义
│   │   ├── models/       # SQLAlchemy ORM 模型
│   │   ├── schemas/      # Pydantic 请求/响应 schema
│   │   ├── services/     # 业务逻辑层
│   │   ├── tasks/        # Celery task 定义
│   │   └── core/         # 配置、数据库连接、Redis 客户端
│   ├── alembic/          # 数据库迁移脚本
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/   # React 组件
│   │   ├── pages/        # 页面
│   │   ├── hooks/        # 自定义 hooks（含 useSSE）
│   │   ├── api/          # API 请求封装
│   │   └── store/        # 状态管理
│   └── public/
├── docker-compose.yml    # PostgreSQL + Redis + Backend + Celery
└── docs/
    └── superpowers/specs/
        └── 2026-04-24-finbot-design.md
```

---

## 10. 关键约定

1. **金额单位：** 所有 API 请求和响应中，金额字段均以"分（fen）"为单位（BIGINT），前端负责展示转换
2. **时区：** 所有时间戳存储为 UTC，前端根据用户时区转换展示
3. **分页：** 列表接口统一使用 `?page=1&size=20`，响应包含 `total` 字段
4. **软删除：** 交易记录不物理删除，添加 `deleted_at` 字段标记
5. **SSE 格式：** `data: {"step":"...","status":"running|done|error","message":"..."}\n\n`
