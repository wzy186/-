package com.dss.ai.tool;

import com.dss.dto.Result;
import com.dss.entity.RedPacket;
import com.dss.service.IRedPacketService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 工具:查询进行中的红包雨场次。
 */
@Component
@RequiredArgsConstructor
public class ListRedPacketsTool implements AiTool {

    private final IRedPacketService redPacketService;

    @Override
    public String name() { return "list_redpackets"; }

    @Override
    public String description() {
        return "查询进行中的红包雨场次。用户问'有什么红包可以抢'时调用,返回场次ID、标题、剩余个数。";
    }

    @Override
    public Map<String, Object> parameters() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        schema.put("properties", new LinkedHashMap<>());
        return schema;
    }

    @Override
    @SuppressWarnings("unchecked")
    public String execute(Map<String, Object> args) {
        Result<?> r = redPacketService.listRedPackets();
        List<RedPacket> list = (List<RedPacket>) r.getData();
        if (list == null || list.isEmpty()) return "暂无红包雨场次";
        List<RedPacket> active = list.stream()
                .filter(rp -> rp.getStatus() != null && rp.getStatus() == 1)
                .toList();
        if (active.isEmpty()) return "暂无进行中的红包雨";
        return active.stream().map(rp -> String.format(
                "- 场次ID:%d,标题:%s,剩余%d个,总额%.2f元",
                rp.getId(), rp.getTitle(),
                rp.getRemainCount() == null ? 0 : rp.getRemainCount(),
                rp.getTotalAmount() == null ? 0 : rp.getTotalAmount() / 100.0
        )).collect(Collectors.joining("\n"));
    }
}
