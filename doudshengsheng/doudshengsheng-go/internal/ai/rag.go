package ai

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"

	"doudshengsheng-go/internal/model"
	"doudshengsheng-go/internal/utils"
)

// RagService RAG 语义问答
type RagService struct {
	emb     *EmbeddingClient
	llm     *LLMClient
	mu      sync.RWMutex
	vectors map[string][]float32 // docID -> 向量
	docs    map[string]string    // docID -> 文本
}

func NewRagService(emb *EmbeddingClient, llm *LLMClient) *RagService {
	return &RagService{
		emb:     emb,
		llm:     llm,
		vectors: map[string][]float32{},
		docs:    map[string]string{},
	}
}

// Reindex 重建索引:商铺 + 笔记向量化
func (r *RagService) Reindex(ctx context.Context) (int, error) {
	r.mu.Lock()
	r.vectors = map[string][]float32{}
	r.docs = map[string]string{}
	r.mu.Unlock()

	count := 0
	// 商铺
	var shops []model.Shop
	utils.DB.Limit(100).Find(&shops)
	for _, s := range shops {
		text := fmt.Sprintf("商铺:%s。地址:%s。均价%d元/人。评分%.1f。销量%d。",
			s.Name, s.Address, s.AvgPrice/100, float64(s.Score)/20.0, s.Sold)
		docID := fmt.Sprintf("shop:%d", s.ID)
		vec, err := r.emb.Embed(text)
		if err != nil {
			log.Printf("商铺 %d 向量化失败: %v", s.ID, err)
			continue
		}
		r.mu.Lock()
		r.vectors[docID] = vec
		r.docs[docID] = text
		r.mu.Unlock()
		count++
	}
	// 笔记
	var blogs []model.Blog
	utils.DB.Limit(100).Find(&blogs)
	for _, b := range blogs {
		text := fmt.Sprintf("探店笔记:%s。内容:%s。点赞%d。", b.Title, b.Content, b.Liked)
		docID := fmt.Sprintf("blog:%d", b.ID)
		vec, err := r.emb.Embed(text)
		if err != nil {
			continue
		}
		r.mu.Lock()
		r.vectors[docID] = vec
		r.docs[docID] = text
		r.mu.Unlock()
		count++
	}
	log.Printf("RAG 索引完成,共 %d 条", count)
	return count, nil
}

// Ask RAG 问答:向量检索 + LLM 生成
func (r *RagService) Ask(ctx context.Context, query string) (string, error) {
	r.mu.RLock()
	empty := len(r.vectors) == 0
	r.mu.RUnlock()
	if empty {
		r.Reindex(ctx)
	}

	// 问题向量化
	qVec, err := r.emb.Embed(query)
	if err != nil {
		// 向量不可用,降级关键词
		return r.askByKeyword(ctx, query)
	}

	// 检索 top5
	r.mu.RLock()
	type sim struct {
		docID string
		score float64
	}
	sims := []sim{}
	for docID, vec := range r.vectors {
		sims = append(sims, sim{docID, Cosine(qVec, vec)})
	}
	r.mu.RUnlock()
	sort.Slice(sims, func(i, j int) bool { return sims[i].score > sims[j].score })

	var sb strings.Builder
	for i := 0; i < 5 && i < len(sims); i++ {
		sb.WriteString("- " + r.docs[sims[i].docID] + "\n")
	}

	prompt := fmt.Sprintf(`你是兜省省的省钱助手。根据以下检索到的商铺/笔记信息回答用户问题。
如果信息不足,如实说明。不要编造未在资料中出现的商铺。

【参考资料】
%s

【用户问题】
%s`, sb.String(), query)

	result, err := r.llm.Chat([]Message{
		{Role: "system", Content: "你是省钱助手,基于资料如实回答。"},
		{Role: "user", Content: prompt},
	}, nil)
	if err != nil {
		return "", err
	}
	return result.Content, nil
}

// askByKeyword 降级:关键词检索
func (r *RagService) askByKeyword(ctx context.Context, query string) (string, error) {
	var shops []model.Shop
	utils.DB.Where("name LIKE ?", "%"+query+"%").Limit(5).Find(&shops)
	var sb strings.Builder
	for _, s := range shops {
		sb.WriteString(fmt.Sprintf("- 商铺:%s。地址:%s。均价%d元/人。\n", s.Name, s.Address, s.AvgPrice/100))
	}
	prompt := fmt.Sprintf(`根据资料回答:【资料】%s【问题】%s`, sb.String(), query)
	result, err := r.llm.Chat([]Message{{Role: "user", Content: prompt}}, nil)
	if err != nil {
		return "", err
	}
	return result.Content, nil
}

// DocCount 文档数
func (r *RagService) DocCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.vectors)
}
