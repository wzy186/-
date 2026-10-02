package com.dss.service;

import com.dss.dto.Result;

public interface IFollowService {
    Result follow(Long followUserId, Boolean isFollow);
    Result isFollow(Long followUserId);
    Result commonFollows(Long targetUserId);
    /** 推模式:发笔记时推送到粉丝收件箱(Feed 流) */
    void pushToFollowers(Long blogId, Long authorId);
    /** 查收件箱:滚动分页 */
    Result feed(Long max, Integer offset);
    /** 用户主页信息:关注数、粉丝数、是否已关注 */
    Result userProfile(Long userId);
}
