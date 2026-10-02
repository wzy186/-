package com.dss.config;

import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.springframework.boot.autoconfigure.jackson.Jackson2ObjectMapperBuilderCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Jackson 全局配置:
 * 1. Long → String,避免雪花 ID(超 2^53)传到前端 JS 丢精度
 * 2. 保留 JavaTimeModule 支持 LocalDateTime
 *
 * 注意:用 modulesToInstall 追加,不要用 modules 覆盖,否则会冲掉 SpringBoot 默认的 JavaTimeModule,
 * 导致 LocalDateTime 序列化报 500。
 */
@Configuration
public class JacksonConfig {

    @Bean
    public Jackson2ObjectMapperBuilderCustomizer longToStringCustomizer() {
        return builder -> {
            SimpleModule longModule = new SimpleModule();
            longModule.addSerializer(Long.class, ToStringSerializer.instance);
            longModule.addSerializer(Long.TYPE, ToStringSerializer.instance);
            builder.modulesToInstall(longModule, new JavaTimeModule());
        };
    }
}
