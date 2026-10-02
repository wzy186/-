package ai

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// LLMClient 调 DeepSeek(OpenAI 兼容格式)
type LLMClient struct {
	apiKey  string
	baseURL string
	model   string
	client  *http.Client
}

func NewLLMClient(apiKey, baseURL, model string) *LLMClient {
	return &LLMClient{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		client:  &http.Client{Timeout: 120 * time.Second},
	}
}

// Message OpenAI 消息格式
type Message struct {
	Role       string                 `json:"role,omitempty"`
	Content    string                 `json:"content,omitempty"`
	ToolCalls  []map[string]interface{} `json:"tool_calls,omitempty"`
	ToolCallID string                 `json:"tool_call_id,omitempty"`
	Name       string                 `json:"name,omitempty"`
}

// ToolDef 工具定义(Function Calling)
type ToolDef struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}
type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ChatResult LLM 返回
type ChatResult struct {
	Content   string                   // 文本回复
	ToolCalls []map[string]interface{} // 工具调用
}

// Chat 同步对话
func (c *LLMClient) Chat(messages []Message, tools []ToolDef) (*ChatResult, error) {
	body := map[string]interface{}{
		"model":       c.model,
		"messages":    messages,
		"temperature": 0.3,
	}
	if len(tools) > 0 {
		body["tools"] = tools
		body["tool_choice"] = "auto"
	}
	bodyBytes, _ := json.Marshal(body)

	req, _ := http.NewRequest("POST", c.baseURL+"/chat/completions", bytes.NewReader(bodyBytes))
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("LLM 调用失败: %w", err)
	}
	defer resp.Body.Close()
	respBytes, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("LLM 返回 %d: %s", resp.StatusCode, string(respBytes))
	}

	var respData struct {
		Choices []struct {
			Message struct {
				Content      string                   `json:"content"`
				ToolCalls    []map[string]interface{} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	json.Unmarshal(respBytes, &respData)
	if len(respData.Choices) == 0 {
		return nil, fmt.Errorf("LLM 无返回")
	}
	return &ChatResult{
		Content:   respData.Choices[0].Message.Content,
		ToolCalls: respData.Choices[0].Message.ToolCalls,
	}, nil
}

// HasToolCall 是否要调工具
func (r *ChatResult) HasToolCall() bool {
	return len(r.ToolCalls) > 0
}

// ChatStream 流式对话(SSE):逐 token 回调
func (c *LLMClient) ChatStream(messages []Message, onToken func(string)) error {
	body := map[string]interface{}{
		"model":       c.model,
		"messages":    messages,
		"temperature": 0.3,
		"stream":      true,
	}
	bodyBytes, _ := json.Marshal(body)

	req, _ := http.NewRequest("POST", c.baseURL+"/chat/completions", bytes.NewReader(bodyBytes))
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(data), &chunk) == nil && len(chunk.Choices) > 0 {
			if chunk.Choices[0].Delta.Content != "" {
				onToken(chunk.Choices[0].Delta.Content)
			}
		}
	}
	return scanner.Err()
}
