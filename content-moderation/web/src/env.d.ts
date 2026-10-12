/// <reference types="vite/client" />

// 兜底声明：element-plus 的 locale 入口若解析不到自带类型时使用
declare module 'element-plus/es/locale/lang/zh-cn' {
  const zhCn: any
  export default zhCn
}
