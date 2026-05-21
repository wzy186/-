package com.mindguard.controller;

import com.mindguard.service.DocumentService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

@RestController
@RequestMapping("/api/knowledge")
@RequiredArgsConstructor
public class AdminController {

    private final DocumentService documentService;

    /**
     * 上传知识库文档（仅管理员）
     */
    @PostMapping("/upload")
    public ResponseEntity<?> uploadDocument(@RequestParam("file") MultipartFile file) {
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "文件不能为空"));
        }

        try {
            // 保存文件到知识库目录
            String filename = file.getOriginalFilename();
            Path dest = Path.of("./knowledge", filename);
            Files.createDirectories(dest.getParent());
            file.transferTo(dest.toFile());

            // 导入到 Chroma
            int chunks = documentService.ingestDocument(dest.toFile());
            return ResponseEntity.ok(Map.of("message", "文档导入成功", "chunks", chunks));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(Map.of("error", "文件上传失败: " + e.getMessage()));
        }
    }

    /**
     * 重新加载整个知识库
     */
    @PostMapping("/reload")
    public ResponseEntity<?> reloadKnowledgeBase() {
        int totalChunks = documentService.autoIngestKnowledgeBase();
        return ResponseEntity.ok(Map.of("message", "知识库重载完成", "totalChunks", totalChunks));
    }

    /**
     * 测试知识库检索
     */
    @GetMapping("/search")
    public ResponseEntity<?> searchKnowledge(@RequestParam String query) {
        var results = documentService.searchKnowledge(query, 5);
        return ResponseEntity.ok(Map.of("results", results));
    }
}
