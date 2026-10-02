package com.dss.controller;

import com.dss.dto.Result;
import com.dss.entity.Blog;
import com.dss.service.IBlogService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/blog")
@RequiredArgsConstructor
public class BlogController {

    private final IBlogService blogService;

    @GetMapping("/hot")
    public Result hot(@RequestParam(value = "current", defaultValue = "1") Integer current) {
        return blogService.queryHotBlog(current);
    }

    @GetMapping("/{id}")
    public Result queryById(@PathVariable Long id) {
        return blogService.queryBlogById(id);
    }

    @PutMapping("/like/{id}")
    public Result like(@PathVariable Long id) {
        return blogService.likeBlog(id);
    }

    @GetMapping("/likes/{id}")
    public Result likes(@PathVariable Long id) {
        return blogService.queryBlogLikes(id);
    }

    /** 按用户查笔记(个人主页用) */
    @GetMapping("/of/user/{userId}")
    public Result ofUser(@PathVariable Long userId) {
        return blogService.queryBlogByUserId(userId);
    }

    @PostMapping
    public Result save(@RequestBody Blog blog) {
        return blogService.saveBlog(blog);
    }
}
