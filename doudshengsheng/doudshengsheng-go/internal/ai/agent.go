package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"doudshengsheng-go/internal/utils"
)

// Agent ReAct 循环:LLM 自主调工具、多步规划
type Agent struct {
	llm   *LLMClient
	tools *ToolRegistry
}

const maxSteps = 6

func NewAgent(llm *LLMClient, tools *ToolRegistry) *Agent {
	return &Agent{llm: llm, tools: tools}
}

// Run 执行 Agent
// onStep: 每步(思考/工具调用/结果)回调
// onToken: 最终答案流式 token 回调
func (a *Agent) Run(ctx context.Context, query string, userID int64, onStep func(string), onToken func(string)) {
	systemPrompt := fmt.Sprintf(`你是兜省省的 AI 省钱助手。用户当前登录 ID: %d。
你可以调用工具查询商铺、优惠券、红包雨,甚至帮用户秒杀下单、抢红包。
规则:
1. 涉及实时数据(商铺/券/红包)时,必须调工具查,不要凭空编造。
2. 帮用户下单/抢红包前,先告知用户你将操作,并确认(除非用户明确说"直接帮我抢")。
3. 回答简洁友好,用中文。
4. 推荐时给出具体商铺名和价格,不要泛泛而谈。`, userID)

	messages := []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: query},
	}
	tools := a.tools.AllDefs()

	for step := 1; step <= maxSteps; step++ {
		result, err := a.llm.Chat(messages, tools)
		if err != nil {
			onStep("⚠️ LLM 调用失败: " + err.Error())
			return
		}

		if result.HasToolCall() {
			// 把 assistant 的 tool_calls 加入历史
			messages = append(messages, Message{
				Role:      "assistant",
				Content:   result.Content,
				ToolCalls: result.ToolCalls,
			})
			// 逐个执行工具
			for _, tc := range result.ToolCalls {
				fn, _ := tc["function"].(map[string]interface{})
				toolName, _ := fn["name"].(string)
				argsJSON, _ := fn["arguments"].(string)
				toolCallID, _ := tc["id"].(string)
				args := parseArgs(argsJSON)
				onStep("🔧 调用工具: " + toolName + " 参数: " + argsJSON)
				toolResult := a.tools.Execute(ctx, toolName, args)
				onStep("📋 结果: " + truncate(toolResult, 200))
				messages = append(messages, Message{
					Role: "tool", Name: toolName, ToolCallID: toolCallID, Content: toolResult,
				})
			}
			continue
		}

		// 没有工具调用 = 最终答案
		answer := result.Content
		if answer == "" {
			answer = "(无回复)"
		}
		onStep("💡 最终回答")
		// 流式输出(这里用分段模拟)
		for _, ch := range answer {
			onToken(string(ch))
		}
		return
	}
	onStep("⚠️ 达到最大步数 " + fmt.Sprintf("%d", maxSteps) + ",停止")
}

func parseArgs(jsonStr string) map[string]interface{} {
	if jsonStr == "" {
		return map[string]interface{}{}
	}
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &m); err != nil {
		return map[string]interface{}{}
	}
	return m
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

var _ = strings.TrimSpace
var _ = utils.GetUserID
