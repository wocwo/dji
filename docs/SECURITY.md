# 安全与多用户隔离

## 认证

支持两种配置：

```env
# 多用户：用户名 -> 密码
AGENT_USERS={"dji":"change-me","alice":"alice-pass","bob":"bob-pass"}

# 兼容单用户 / CI：密码作为 dji 的密码
AGENT_ACCESS_TOKEN=change-me
```

- 配置了 `AGENT_USERS` 或 `AGENT_ACCESS_TOKEN` 后，所有接口要求 Basic Auth。
- `dji` 是默认管理员用户名；如果 `AGENT_USERS` 已包含 `dji`，以用户表为准。
- 支持 `Authorization: Basic`、`Authorization: Bearer`、HttpOnly 会话 Cookie。
- **不再接受 URL query 里的 `access_token`**，避免 token 出现在日志、Referer 和浏览器历史。
- 密码只从环境变量读取，不写入代码。

## 会话隔离

每个用户的 session 在 SQLite 中使用命名空间：

```text
u:<user>:<frontend-session-id>
```

API 行为：

- `POST /v1/agent/jobs` 用当前用户的命名空间写入；
- `GET /v1/sessions` 只返回当前用户前缀的会话；
- `GET /v1/sessions/{id}` 只能读到自己的 turn / summary；
- job 记录 `owner`，其他用户读取 job 返回 404，而不是泄露存在性；
- 限流按用户计数（未认证模式按 IP）。

测试：`tests/test_multiuser.py` 覆盖 alice/bob 同名 session、job 越权、turn 隔离。

## 当前边界

这是**单进程、单 SQLite** 的轻量隔离，不是生产级多租户：

- 密码是环境变量明文，生产应改 bcrypt/Argon2 或 OIDC；
- job 队列在内存里，进程重启会丢失；
- 没有审计日志、配额、租户级数据删除；
- 没有 TLS，生产必须在反向代理后启用 HTTPS；
- 上传图片不落盘；视觉请求会发送给 DashScope，需要用户知情；
- `.env.agent` 含 Key，不能提交；本仓库 `.gitignore` 已忽略。

## 上线检查清单

- [ ] 轮换所有在聊天/截图里出现过的 Key
- [ ] 使用反向代理 + HTTPS
- [ ] 设置独立 `AGENT_USERS`，不要共用 `dji` 密码
- [ ] 关闭公网 quick tunnel，使用固定域名
- [ ] 限制请求体大小和每分钟请求数
- [ ] 设置图片保留策略（当前不落盘）
- [ ] 备份 SQLite，验证恢复
- [ ] 在隐私政策中说明第三方视觉模型
