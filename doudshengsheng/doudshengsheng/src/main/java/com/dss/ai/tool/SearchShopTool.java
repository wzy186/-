package com.dss.ai.tool;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.dss.entity.Shop;
import com.dss.entity.ShopType;
import com.dss.mapper.ShopMapper;
import com.dss.mapper.ShopTypeMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 工具:搜索商铺。
 * LLM 可调:用户问"有什么奶茶店"→ LLM 调此工具查 DB。
 */
@Component
@RequiredArgsConstructor
public class SearchShopTool implements AiTool {

    private final ShopMapper shopMapper;
    private final ShopTypeMapper shopTypeMapper;

    @Override
    public String name() { return "search_shops"; }

    @Override
    public String description() {
        return "搜索商铺。可按类型(美食/娱乐/丽人/生活服务/酒店)或关键词搜索。返回商铺名、地址、均价、评分、销量。";
    }

    @Override
    public Map<String, Object> parameters() {
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("type", Map.of("type", "string", "description", "商铺类型,如:美食、娱乐、丽人、生活服务、酒店。可选"));
        props.put("keyword", Map.of("type", "string", "description", "商铺名关键词,如:奶茶、火锅。可选"));
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        schema.put("properties", props);
        return schema;
    }

    @Override
    public String execute(Map<String, Object> args) {
        String type = (String) args.get("type");
        String keyword = (String) args.get("keyword");

        LambdaQueryWrapper<Shop> qw = new LambdaQueryWrapper<>();
        if (keyword != null && !keyword.isBlank()) {
            qw.like(Shop::getName, keyword);
        }
        if (type != null && !type.isBlank()) {
            // 类型名 → id
            List<ShopType> types = shopTypeMapper.selectList(null);
            ShopType matched = types.stream().filter(t -> t.getName().contains(type)).findFirst().orElse(null);
            if (matched != null) qw.eq(Shop::getTypeId, matched.getId());
        }
        qw.last("limit 10");
        List<Shop> shops = shopMapper.selectList(qw);
        if (shops.isEmpty()) return "没有找到匹配的商铺";
        return shops.stream().map(s -> String.format(
                "- %s(地址:%s,均价%d元/人,评分%.1f,销量%d)",
                s.getName(), s.getAddress() == null ? "未知" : s.getAddress(),
                s.getAvgPrice() == null ? 0 : s.getAvgPrice() / 100,
                s.getScore() == null ? 0 : s.getScore() / 20.0,
                s.getSold() == null ? 0 : s.getSold()
        )).collect(Collectors.joining("\n"));
    }
}
