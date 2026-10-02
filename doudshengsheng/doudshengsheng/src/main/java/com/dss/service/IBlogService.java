package com.dss.service;

import com.dss.dto.Result;
import com.dss.entity.Blog;

public interface IBlogService {
    Result queryHotBlog(Integer current);
    /** 笔记详情:含作者信息 + 当前用户是否已赞 */
    Result queryBlogById(Long id);
    Result likeBlog(Long id);
    Result queryBlogLikes(Long id);
    Result saveBlog(Blog blog);
    /** 按用户查笔记(个人主页用) */
    Result queryBlogByUserId(Long userId);
}
