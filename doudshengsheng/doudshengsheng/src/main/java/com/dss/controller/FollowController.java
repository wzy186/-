package com.dss.controller;

import com.dss.dto.Result;
import com.dss.service.IFollowService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/follow")
@RequiredArgsConstructor
public class FollowController {

    private final IFollowService followService;

    @PutMapping("/{id}/{isFollow}")
    public Result follow(@PathVariable Long id, @PathVariable Boolean isFollow) {
        return followService.follow(id, isFollow);
    }

    @GetMapping("/or/{id}")
    public Result isFollow(@PathVariable Long id) {
        return followService.isFollow(id);
    }

    @GetMapping("/common/{id}")
    public Result commonFollows(@PathVariable Long id) {
        return followService.commonFollows(id);
    }

    /** 用户主页:关注数/粉丝数/是否已关注 */
    @GetMapping("/profile/{id}")
    public Result profile(@PathVariable Long id) {
        return followService.userProfile(id);
    }

    /**
     * Feed 流收件箱(滚动分页)
     * max: 上次返回的 minTime;offset: 上次返回的 offset
     */
    @GetMapping("/feed")
    public Result feed(@RequestParam(value = "max", required = false) Long max,
                       @RequestParam(value = "offset", required = false) Integer offset) {
        return followService.feed(max, offset);
    }
}
