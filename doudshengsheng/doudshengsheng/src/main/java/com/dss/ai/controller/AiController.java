package com.dss.ai.controller;

import com.dss.ai.agent.Agent;
import com.dss.ai.rag.RagService;
import com.dss.dto.Result;
import com.dss.utils.UserHolder;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * AI 助手接口。
 * - /ai/chat:Agent 多步规划(SSE 流式,含工具调用过程)
 * - /ai/rag:RAG 语义问答(商铺/笔记检索 + LLM)
 */
@Tag(name = "AI 助手", description = "Agent 多步规划 + RAG 语义问答")
@Slf4j
@RestController
@RequestMapping("/ai")
@RequiredArgsConstructor
public class AiController {

    private final Agent agent;
    private final RagService ragService;
    private final ExecutorService executor = Executors.newCachedThreadPool();

    /**
     * Agent 对话(SSE 流式)。
     * 前端用 EventSource 接收,事件类型:
     *   - step: 思考/工具调用步骤
     *   - token: 最终答案的字符(打字机)
     *   - done: 结束
     *   - error: 异常
     */
    @Operation(summary = "Agent 对话(流式)", description = "LLM 自主调工具查商铺/抢红包/秒杀,多步规划")
    @PostMapping(value = "/chat", produces = "text/event-stream")
    public SseEmitter chat(@RequestParam("query") String query) {
        Long userId = UserHolder.getUserId();
        SseEmitter emitter = new SseEmitter(120_000L);

        executor.submit(() -> {
            try {
                send(emitter, "step", "🤔 思考中: " + query);
                agent.run(query, userId, step -> send(emitter, "step", step), token -> send(emitter, "token", token));
                send(emitter, "done", "完成");
                emitter.complete();
            } catch (Exception e) {
                log.error("Agent 异常", e);
                send(emitter, "error", e.getMessage());
                emitter.completeWithError(e);
            }
        });
        return emitter;
    }

    /**
     * RAG 语义问答(同步,返回完整答案)。
     * 用户问"附近哪里奶茶便宜"→ 向量检索商铺/笔记 → LLM 基于检索结果回答。
     */
    @Operation(summary = "RAG 语义问答", description = "向量检索商铺/笔记 + LLM 生成答案")
    @PostMapping("/rag")
    public Result rag(@RequestParam("query") String query) {
        try {
            String answer = ragService.ask(query);
            return Result.ok(answer);
        } catch (Exception e) {
            log.error("RAG 异常", e);
            return Result.fail("AI 问答失败: " + e.getMessage());
        }
    }

    /**
     * 重建向量索引(管理员用,商铺/笔记变更后调用)。
     */
    @Operation(summary = "重建 RAG 向量索引")
    @PostMapping("/reindex")
    public Result reindex() {
        try {
            int n = ragService.reindex();
            return Result.ok(Map.of("indexed", n));
        } catch (Exception e) {
            return Result.fail("重建失败: " + e.getMessage());
        }
    }

    private void send(SseEmitter emitter, String event, String data) {
        try {
            emitter.send(SseEmitter.event().name(event).data(data == null ? "" : data));
        } catch (IOException e) {
            log.debug("SSE 发送失败(客户端可能已断开): {}", e.getMessage());
        }
    }
}
