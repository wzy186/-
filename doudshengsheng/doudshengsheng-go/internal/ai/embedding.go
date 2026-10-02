package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
)

// EmbeddingClient 调本地 ollama 把文本转向量
type EmbeddingClient struct {
	baseURL string
	model   string
	client  *http.Client
}

func NewEmbeddingClient(baseURL, model string) *EmbeddingClient {
	return &EmbeddingClient{
		baseURL: baseURL,
		model:   model,
		client:  &http.Client{Timeout: 60 * time.Second},
	}
}

// Embed 文本转向量
func (c *EmbeddingClient) Embed(text string) ([]float32, error) {
	if strings.TrimSpace(text) == "" {
		return nil, nil
	}
	if len(text) > 2000 {
		text = text[:2000]
	}
	body, _ := json.Marshal(map[string]string{"model": c.model, "input": text})
	req, _ := http.NewRequest("POST", c.baseURL+"/api/embed", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBytes, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		// 回退旧接口
		return c.embedLegacy(text)
	}

	// 新版 ollama:embeddings 数组嵌套
	var data struct {
		Embeddings [][]float32 `json:"embeddings"`
		Embedding  []float32   `json:"embedding"`
	}
	json.Unmarshal(respBytes, &data)
	if len(data.Embeddings) > 0 {
		return data.Embeddings[0], nil
	}
	if len(data.Embedding) > 0 {
		return data.Embedding, nil
	}
	return c.embedLegacy(text)
}

// embedLegacy 回退旧接口 /api/embeddings
func (c *EmbeddingClient) embedLegacy(text string) ([]float32, error) {
	body, _ := json.Marshal(map[string]string{"model": c.model, "input": text})
	req, _ := http.NewRequest("POST", c.baseURL+"/api/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBytes, _ := io.ReadAll(resp.Body)
	var data struct {
		Embedding []float32 `json:"embedding"`
	}
	json.Unmarshal(respBytes, &data)
	if len(data.Embedding) == 0 {
		return nil, fmt.Errorf("embedding 为空")
	}
	return data.Embedding, nil
}

// Cosine 余弦相似度
func Cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
