package com.dss.ai.tool;

import com.dss.dto.Result;
import com.dss.service.IVoucherOrderService;
import com.dss.utils.UserHolder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 工具:秒杀下单。
 * 注意:用当前登录用户(UserHolder),不暴露给 LLM 任意用户。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class SeckillTool implements AiTool {

    private final IVoucherOrderService voucherOrderService;

    @Override
    public String name() { return "seckill_voucher"; }

    @Override
    public String description() {
        return "对秒杀券下单(秒杀)。需要券ID。用户说'帮我秒杀/抢电影票'时调用。";
    }

    @Override
    public Map<String, Object> parameters() {
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("voucherId", Map.of("type", "integer", "description", "秒杀券ID,如 10 或 23"));
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        schema.put("properties", props);
        schema.put("required", java.util.List.of("voucherId"));
        return schema;
    }

    @Override
    public String execute(Map<String, Object> args) {
        Long userId = UserHolder.getUserId();
        if (userId == null) return "需要先登录";
        Object idObj = args.get("voucherId");
        if (idObj == null) return "缺少参数: voucherId";
        Long voucherId = Long.valueOf(idObj.toString());
        Result<?> r = voucherOrderService.seckillVoucher(voucherId);
        if (r.getSuccess()) {
            return "秒杀成功!订单号: " + r.getData();
        }
        return "秒杀失败: " + r.getMsg();
    }
}
