package com.dss.ai.tool;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.dss.entity.Voucher;
import com.dss.mapper.VoucherMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 工具:查询优惠券(含秒杀券)。
 */
@Component
@RequiredArgsConstructor
public class ListVouchersTool implements AiTool {

    private final VoucherMapper voucherMapper;

    @Override
    public String name() { return "list_vouchers"; }

    @Override
    public String description() {
        return "查询所有优惠券(含秒杀券)。返回券名、抵扣金额、是否秒杀券。用户问有什么优惠时调用。";
    }

    @Override
    public Map<String, Object> parameters() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        schema.put("properties", new LinkedHashMap<>());
        return schema;
    }

    @Override
    public String execute(Map<String, Object> args) {
        List<Voucher> list = voucherMapper.selectList(new LambdaQueryWrapper<Voucher>().last("limit 20"));
        if (list.isEmpty()) return "暂无优惠券";
        return list.stream().map(v -> String.format(
                "- %s(%s,抵扣%d元%s)",
                v.getTitle(), v.getSubTitle() == null ? "" : v.getSubTitle(),
                v.getActualValue() == null ? 0 : v.getActualValue() / 100,
                v.getType() != null && v.getType() == 2 ? ",秒杀券" : ""
        )).collect(Collectors.joining("\n"));
    }
}
