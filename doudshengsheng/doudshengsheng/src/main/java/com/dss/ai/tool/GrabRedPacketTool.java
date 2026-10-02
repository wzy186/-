package com.dss.ai.tool;

import com.dss.dto.Result;
import com.dss.service.IRedPacketService;
import com.dss.utils.UserHolder;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 工具:抢红包。
 */
@Component
@RequiredArgsConstructor
public class GrabRedPacketTool implements AiTool {

    private final IRedPacketService redPacketService;

    @Override
    public String name() { return "grab_redpacket"; }

    @Override
    public String description() {
        return "抢红包雨。需要红包场次ID。用户说'帮我抢红包'时调用。";
    }

    @Override
    public Map<String, Object> parameters() {
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("redPacketId", Map.of("type", "integer", "description", "红包场次ID"));
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        schema.put("properties", props);
        schema.put("required", java.util.List.of("redPacketId"));
        return schema;
    }

    @Override
    public String execute(Map<String, Object> args) {
        Long userId = UserHolder.getUserId();
        if (userId == null) return "需要先登录";
        Object idObj = args.get("redPacketId");
        if (idObj == null) return "缺少参数: redPacketId";
        Long rpId = Long.valueOf(idObj.toString());
        Result<?> r = redPacketService.grab(rpId);
        if (r.getSuccess()) {
            return "抢到红包!金额: " + (Long.valueOf(r.getData().toString()) / 100.0) + " 元";
        }
        return "抢红包失败: " + r.getMsg();
    }
}
