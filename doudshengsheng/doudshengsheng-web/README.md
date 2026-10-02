# 兜省省 前端 (doudshengsheng-web)

Vue3 + Vite + Element Plus + Axios + Vue Router。支持双后端模式。

## 页面

- C 端 12 个：登录、商铺列表/详情、附近商铺、优惠券秒杀、红包雨、探店笔记/详情、个人主页、关注/Feed、签到/UV、AI 省钱助手
- 商户后台 3 个（`/admin`，仅 role=1）：数据概览、秒杀券管理、红包雨数据

## 启动

```bash
npm install
npm run dev      # 5173 → 代理 Java 后端 http://localhost:8081
npm run dev:go   # 5174 → 代理 Go 后端 http://localhost:8082
```

## 说明

- `vite.config.js` 把 `/api/*` 代理到后端并去掉前缀，无跨域问题
- `src/api/index.js` 统一封装 43 个接口；`src/utils/request.js` 注入 Token、统一错误处理、401 跳登录
- 静态封面图在 `public/covers/`，商铺无图时按 `/covers/shop_{id}.svg` 回退
