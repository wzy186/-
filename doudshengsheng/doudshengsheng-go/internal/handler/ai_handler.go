package handler

import (
	"context"
	"fmt"
	"io"

	"doudshengsheng-go/internal/ai"
	"doudshengsheng-go/internal/middleware"
	"doudshengsheng-go/internal/utils"

	"github.com/cloudwego/hertz/pkg/app"
	"github.com/cloudwego/hertz/pkg/protocol/consts"
	"github.com/cloudwego/hertz/pkg/route"
)

type AIHandler struct {
	agent *ai.Agent
	rag   *ai.RagService
}

func NewAIHandler(agent *ai.Agent, rag *ai.RagService) *AIHandler {
	return &AIHandler{agent: agent, rag: rag}
}

func (h *AIHandler) Register(group *route.RouterGroup) {
	aiGroup := group.Group("", middleware.LoginRequired())
	aiGroup.POST("/ai/chat", h.Chat)
	aiGroup.POST("/ai/rag", h.Rag)
	aiGroup.POST("/ai/reindex", h.Reindex)
}

// Chat Agent 对话(SSE 流式)
func (h *AIHandler) Chat(c context.Context, ctx *app.RequestContext) {
	query := string(ctx.Query("query"))
	uid := utils.GetUserID(c)

	// SSE 头
	ctx.SetStatusCode(consts.StatusOK)
	ctx.Response.Header.Set("Content-Type", "text/event-stream")
	ctx.Response.Header.Set("Cache-Control", "no-cache")
	ctx.Response.Header.Set("Connection", "keep-alive")
	ctx.Response.Header.Set("Transfer-Encoding", "chunked")

	// 用 io.Pipe 流式写
	pr, pw := io.Pipe()
	ctx.SetBodyStream(pr, -1)

	go func() {
		defer pw.Close()
		send := func(event, data string) {
			fmt.Fprintf(pw, "event:%s\ndata:%s\n\n", event, data)
		}
		send("step", "🤔 思考中: "+query)
		h.agent.Run(c, query, uid,
			func(step string) { send("step", step) },
			func(token string) { send("token", token) },
		)
		send("done", "完成")
	}()
}

// Rag RAG 问答(同步)
func (h *AIHandler) Rag(c context.Context, ctx *app.RequestContext) {
	query := string(ctx.Query("query"))
	answer, err := h.rag.Ask(c, query)
	if err != nil {
		ctx.JSON(consts.StatusOK, utils.Fail("AI 问答失败: "+err.Error()))
		return
	}
	ctx.JSON(consts.StatusOK, utils.OKWith(answer))
}

// Reindex 重建索引
func (h *AIHandler) Reindex(c context.Context, ctx *app.RequestContext) {
	n, err := h.rag.Reindex(c)
	if err != nil {
		ctx.JSON(consts.StatusOK, utils.Fail("重建失败: "+err.Error()))
		return
	}
	ctx.JSON(consts.StatusOK, utils.OKWith(map[string]int{"indexed": n}))
}
