package com.dss.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * OpenAPI/Swagger 文档配置。
 * 访问:/swagger-ui.html(界面)、/v3/api-docs(JSON)
 */
@Configuration
public class OpenApiConfig {

    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("兜省省 API 文档")
                        .description("模拟抖省省的 Redis 实战后端,含红包雨/秒杀/缓存三大问题等")
                        .version("1.0.0")
                        .contact(new Contact().name("doudshengsheng"))
                        .license(new License().name("MIT")));
    }
}
