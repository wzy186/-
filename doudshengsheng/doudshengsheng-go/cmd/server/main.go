package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"doudshengsheng-go/internal/ai"
	"doudshengsheng-go/internal/cache"
	"doudshengsheng-go/internal/config"
	"doudshengsheng-go/internal/handler"
	"doudshengsheng-go/internal/middleware"
	"doudshengsheng-go/internal/service"
	"doudshengsheng-go/internal/utils"

	"github.com/cloudwego/hertz/pkg/app/server"
	"github.com/cloudwego/hertz/pkg/common/hlog"
)

func main() {
	// 1. 加载配置
	cfg, err := config.Load("config.yaml")
	if err != nil {
		panic("加载配置失败: " + err.Error())
	}

	// 2. 初始化 DB/Redis
	if err := utils.InitDB(cfg.MySQL.DSN); err != nil {
		panic(err)
	}
	if err := utils.InitRedis(cfg.Redis.Addr, cfg.Redis.Password, cfg.Redis.DB); err != nil {
		panic(err)
	}
	hlog.Info("DB/Redis 初始化完成")

	// 3. 组装依赖(Go 没有自动 DI,手动 wire)
	cc := cache.New(utils.Redis)
	userSvc := &service.UserService{}
	shopSvc := service.NewShopService(cc)
	seckillSvc := service.NewSeckillService()
	voucherSvc := &service.VoucherService{}
	rpSvc := service.NewRedPacketService()
	followSvc := &service.FollowService{}
	blogSvc := service.NewBlogService(followSvc)
	statsSvc := &service.StatsService{}

	// AI 模块
	llmClient := ai.NewLLMClient(cfg.AI.Deepseek.APIKey, cfg.AI.Deepseek.BaseURL, cfg.AI.Deepseek.Model)
	embClient := ai.NewEmbeddingClient(cfg.AI.Ollama.BaseURL, cfg.AI.Ollama.EmbeddingModel)
	toolRegistry := ai.NewToolRegistry()
	toolRegistry.Register(ai.SearchShopTool{})
	toolRegistry.Register(ai.ListVouchersTool{})
	toolRegistry.Register(ai.ListRedPacketsTool{})
	toolRegistry.Register(ai.SeckillTool{Svc: seckillSvc})
	toolRegistry.Register(ai.GrabRedPacketTool{Svc: rpSvc})
	agent := ai.NewAgent(llmClient, toolRegistry)
	ragSvc := ai.NewRagService(embClient, llmClient)

	// 4. 启动预热
	ctx := context.Background()
	shopSvc.PreheatBloom(ctx)
	voucherSvc.PreheatAll(ctx)
	seckillSvc.StartConsumer()   // 秒杀订单消费者
	rpSvc.StartConsumer()        // 红包记录消费者

	// RAG 索引(异步,失败不影响启动;需 ollama)
	go func() {
		if _, err := ragSvc.Reindex(ctx); err != nil {
			hlog.Warn("RAG 索引失败(ollama 未就绪?): ", err.Error())
		}
	}()

	// 5. 路由
	h := server.Default(server.WithHostPorts(fmt.Sprintf(":%d", cfg.Server.Port)))
	h.Use(middleware.Recover(), middleware.CORS(), middleware.RefreshToken())

	api := h.Group("")
	{
		userH := handler.NewUserHandler(userSvc)
		userH.Register(api)

		shopH := handler.NewShopHandler(shopSvc)
		shopH.Register(api)

		voucherH := handler.NewVoucherHandler(seckillSvc, voucherSvc)
		voucherH.Register(api)

		rpH := handler.NewRedPacketHandler(rpSvc)
		rpH.Register(api)

		socialH := handler.NewSocialHandler(blogSvc, followSvc, statsSvc)
		socialH.Register(api)

		aiH := handler.NewAIHandler(agent, ragSvc)
		aiH.Register(api)

		docH := handler.NewDocHandler()
		docH.Register(api)
	}

	// 6. 启动
	hlog.Info("兜省省 Go 版启动,端口 ", cfg.Server.Port)
	go h.Spin()

	// 7. 优雅退出
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	hlog.Info("服务关闭中...")
	utils.Redis.Close()
	_ = utils.DB
}
