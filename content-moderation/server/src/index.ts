import { createApp } from './app.js';
import { seedIfEmpty } from './seed.js';

const PORT = Number(process.env.PORT ?? 3000);

const seeded = seedIfEmpty();
if (seeded) console.log('📦 检测到空数据库，已自动写入演示数据');
createApp().listen(PORT, () => {
  console.log(`✅ 审核服务已启动: http://localhost:${PORT}`);
});
